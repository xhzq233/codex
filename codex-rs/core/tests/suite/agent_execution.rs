use anyhow::Result;
use codex_core::config::AgentRoleConfig;
use codex_features::Feature;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::time::Duration;

const FIRST_PROMPT: &str = "spawn the first worker";
const FIRST_TASK: &str = "first worker task";
const SECOND_TASK: &str = "second worker task";
const MULTI_AGENT_V2_NAMESPACE: &str = "collaboration";
const PORTABLE_WORKER_ROLE: &str = "portable_worker";

fn body_contains(request: &wiremock::Request, text: &str) -> bool {
    serde_json::from_slice::<serde_json::Value>(&request.body)
        .is_ok_and(|body| body.to_string().contains(text))
}

fn has_function_call_output(request: &wiremock::Request, call_id: &str) -> bool {
    serde_json::from_slice::<serde_json::Value>(&request.body).is_ok_and(|body| {
        body.get("input")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("type").and_then(serde_json::Value::as_str)
                        == Some("function_call_output")
                        && item.get("call_id").and_then(serde_json::Value::as_str) == Some(call_id)
                })
            })
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_nested_spawn_checks_shared_active_execution_capacity() -> Result<()> {
    let server = start_mock_server().await;
    let first_args = serde_json::to_string(&json!({
        "message": FIRST_TASK,
        "task_name": "first",
    }))?;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, FIRST_PROMPT),
        sse(vec![
            ev_response_created("first-response"),
            ev_function_call_with_namespace(
                "first-call",
                MULTI_AGENT_V2_NAMESPACE,
                "spawn_agent",
                &first_args,
            ),
            ev_completed("first-response"),
        ]),
    )
    .await;
    let second_args = serde_json::to_string(&json!({
        "message": SECOND_TASK,
        "task_name": "second",
    }))?;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, FIRST_TASK) && !has_function_call_output(request, "first-call")
        },
        sse(vec![
            ev_response_created("first-worker-response"),
            ev_function_call_with_namespace(
                "second-call",
                MULTI_AGENT_V2_NAMESPACE,
                "spawn_agent",
                &second_args,
            ),
            ev_completed("first-worker-response"),
        ]),
    )
    .await;
    let second_followup = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| has_function_call_output(request, "second-call"),
        sse(vec![
            ev_response_created("second-followup-response"),
            ev_assistant_message("second-followup-message", "blocked"),
            ev_completed("second-followup-response"),
        ]),
    )
    .await;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| has_function_call_output(request, "first-call"),
        sse(vec![
            ev_response_created("first-followup-response"),
            ev_assistant_message("first-followup-message", "spawned"),
            ev_completed("first-followup-response"),
        ]),
    )
    .await;

    let mut builder = test_codex().with_model("koffing").with_config(|config| {
        config
            .features
            .enable(Feature::Collab)
            .expect("test config should allow feature update");
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("test config should allow feature update");
        config.multi_agent_v2.max_concurrent_threads_per_session = 2;
    });
    let test = builder.build(&server).await?;
    test.submit_turn(FIRST_PROMPT).await?;

    let second_output = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Some(output) = second_followup.function_call_output_text("second-call") {
                return output;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    assert_eq!(
        second_output,
        "collab spawn failed: agent thread limit reached"
    );
    assert_eq!(test.thread_manager.list_thread_ids().await.len(), 2);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_plaintext_inter_agent_messages_use_portable_responses_items() -> Result<()> {
    let server = start_mock_server().await;
    let task = "portable worker task";
    let spawn_args = serde_json::to_string(&json!({
        "message": task,
        "task_name": "portable",
        "agent_type": PORTABLE_WORKER_ROLE,
        "fork_turns": "none",
    }))?;
    let root_response = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, "spawn portable worker")
                && !body_contains(request, "portable worker task")
        },
        sse(vec![
            ev_response_created("root-response"),
            ev_function_call_with_namespace(
                "spawn-call",
                MULTI_AGENT_V2_NAMESPACE,
                "spawn_agent",
                &spawn_args,
            ),
            ev_completed("root-response"),
        ]),
    )
    .await;
    let worker_response = mount_sse_once_match(
        &server,
        move |request: &wiremock::Request| {
            body_contains(request, task) && !has_function_call_output(request, "spawn-call")
        },
        sse(vec![
            ev_response_created("worker-response"),
            ev_assistant_message("worker-message", "done"),
            ev_completed("worker-response"),
        ]),
    )
    .await;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| has_function_call_output(request, "spawn-call"),
        sse(vec![
            ev_response_created("root-followup"),
            ev_assistant_message("root-message", "complete"),
            ev_completed("root-followup"),
        ]),
    )
    .await;

    let worker_base_url = format!("{}/v1", server.uri());
    let test = test_codex()
        .with_model("koffing")
        .with_config(move |config| {
            config
                .features
                .enable(Feature::Collab)
                .expect("test config should allow feature update");
            config
                .features
                .enable(Feature::MultiAgentV2)
                .expect("test config should allow feature update");
            config.multi_agent_v2.max_concurrent_threads_per_session = 2;
            config.multi_agent_v2.encrypt_inter_agent_messages = false;
            let role_path = config.codex_home.join("portable-worker-role.toml");
            std::fs::write(
                &role_path,
                format!(
                    "model = \"koffing\"\nmodel_provider = \"portable\"\n\n[model_providers.portable]\nname = \"portable\"\nbase_url = \"{worker_base_url}\"\nenv_key = \"PATH\"\nwire_api = \"responses\"\n"
                ),
            )
            .expect("write portable worker role config");
            config.agent_roles.insert(
                PORTABLE_WORKER_ROLE.to_string(),
                AgentRoleConfig {
                    description: Some("Portable Responses provider".to_string()),
                    config_file: Some(role_path.to_path_buf()),
                    nickname_candidates: None,
                },
            );
        })
        .build_with_auto_env(&server)
        .await?;
    test.submit_turn("spawn portable worker").await?;

    let root_body = root_response
        .requests()
        .into_iter()
        .next()
        .expect("root request")
        .body_json();
    let spawn_message_schema = root_body["tools"]
        .as_array()
        .and_then(|tools| {
            tools.iter().find_map(|tool| {
                if tool.get("type").and_then(serde_json::Value::as_str) != Some("namespace") {
                    return None;
                }
                tool.get("tools")?.as_array()?.iter().find_map(|child| {
                    (child.get("name").and_then(serde_json::Value::as_str) == Some("spawn_agent"))
                        .then(|| &child["parameters"]["properties"]["message"])
                })
            })
        })
        .expect("spawn_agent message schema");
    assert_eq!(spawn_message_schema.get("encrypted"), None);

    let worker_input = worker_response
        .requests()
        .into_iter()
        .flat_map(|request| request.input())
        .collect::<Vec<_>>();
    assert!(
        worker_input
            .iter()
            .all(|item| item["type"] != "agent_message")
    );
    assert_eq!(
        worker_input
            .into_iter()
            .find(|item| item["content"] == json!([{"type": "input_text", "text": task}])),
        Some(json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": task}],
        }))
    );

    Ok(())
}
