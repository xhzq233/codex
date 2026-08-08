//! Launch MCP stdio servers and return the transport rmcp should use.
//!
//! This module owns the "where does the server process run?" decision:
//!
//! - [`LocalStdioServerLauncher`] starts the configured command as a child of
//!   the orchestrator process.
//! - [`ExecutorStdioServerLauncher`] starts the configured command through the
//!   executor process API.
//!
//! Both paths return [`StdioServerTransport`], so `RmcpClient` can hand the
//! resulting byte stream to rmcp without knowing where the process lives. The
//! executor-specific byte adaptation lives in `executor_process_transport`.

use std::collections::HashMap;
use std::ffi::OsString;
use std::future::Future;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
#[cfg(unix)]
use std::thread::sleep;
#[cfg(unix)]
use std::thread::spawn;
#[cfg(unix)]
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use codex_config::types::McpServerEnvVar;
use codex_exec_server::ExecBackend;
use codex_exec_server::ExecEnvPolicy;
use codex_exec_server::ExecParams;
use codex_exec_server::ExecProcess;
use codex_protocol::config_types::ShellEnvironmentPolicyInherit;
use codex_utils_path_uri::LegacyAppPathString;
use codex_utils_path_uri::PathUri;
#[cfg(unix)]
use codex_utils_pty::process_group::kill_process_group;
#[cfg(unix)]
use codex_utils_pty::process_group::terminate_process_group;
use futures::FutureExt;
use futures::future::BoxFuture;
use rmcp::service::RoleClient;
use rmcp::service::RxJsonRpcMessage;
use rmcp::service::TxJsonRpcMessage;
use rmcp::transport::Transport;
use rmcp::transport::child_process::TokioChildProcess;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Command;
use tracing::info;
use tracing::warn;

use crate::executor_process_transport::ExecutorProcessTransport;
use crate::local_stdio_transport::LocalStdioTransport;
use crate::program_resolver;
use crate::protocol_mode::McpProtocolMode;
use crate::utils::create_env_for_mcp_server;
use crate::utils::create_env_overlay_for_remote_mcp_server;
use crate::utils::remote_mcp_env_var_names;

// General purpose public code.

/// Launches an MCP stdio server and returns the transport for rmcp.
///
/// This trait is the boundary between MCP lifecycle code and process placement.
/// `RmcpClient` owns MCP operations such as `initialize` and `tools/list`; the
/// launcher owns starting the configured command and producing an rmcp
/// [`Transport`] over the server's stdin/stdout bytes.
pub trait StdioServerLauncher: private::Sealed + Send + Sync {
    /// Start the configured stdio server and return its rmcp-facing transport.
    fn launch(
        &self,
        command: StdioServerCommand,
    ) -> BoxFuture<'static, io::Result<StdioServerTransport>>;
}

/// Command-line process shape shared by stdio server launchers.
#[derive(Clone)]
pub struct StdioServerCommand {
    program: OsString,
    args: Vec<OsString>,
    env: Option<HashMap<OsString, OsString>>,
    env_vars: Vec<McpServerEnvVar>,
    cwd: Option<String>,
    protocol_mode: McpProtocolMode,
}

/// Client-side rmcp transport for a launched MCP stdio server.
///
/// The concrete process placement stays private to this module. `RmcpClient`
/// only sees the standard rmcp transport abstraction and can pass this value
/// directly to `rmcp::service::serve_client`.
pub struct StdioServerTransport {
    inner: StdioServerTransportInner,
    process: StdioServerProcessHandle,
}

enum StdioServerTransportInner {
    LocalLegacy(TokioChildProcess),
    LocalModern(LocalStdioTransport),
    Executor(ExecutorProcessTransport),
}

impl Transport<RoleClient> for StdioServerTransport {
    type Error = io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = std::result::Result<(), Self::Error>> + Send + 'static {
        // Both variants already implement rmcp's transport contract. This
        // wrapper keeps process placement private while leaving rmcp's send
        // semantics unchanged.
        match &mut self.inner {
            StdioServerTransportInner::LocalLegacy(transport) => transport.send(item).boxed(),
            StdioServerTransportInner::LocalModern(transport) => transport.send(item).boxed(),
            StdioServerTransportInner::Executor(transport) => transport.send(item).boxed(),
        }
    }

    fn receive(&mut self) -> impl Future<Output = Option<RxJsonRpcMessage<RoleClient>>> + Send {
        // rmcp reads from the same transport shape for both placements. The
        // executor variant turns pushed process-output events back into the
        // line-delimited JSON stream expected by rmcp.
        match &mut self.inner {
            StdioServerTransportInner::LocalLegacy(transport) => transport.receive().boxed(),
            StdioServerTransportInner::LocalModern(transport) => transport.receive().boxed(),
            StdioServerTransportInner::Executor(transport) => transport.receive().boxed(),
        }
    }

    async fn close(&mut self) -> std::result::Result<(), Self::Error> {
        self.process.terminate().await?;
        match &mut self.inner {
            StdioServerTransportInner::LocalLegacy(transport) => transport.close().await,
            StdioServerTransportInner::LocalModern(transport) => transport.close().await,
            StdioServerTransportInner::Executor(transport) => transport.close().await,
        }
    }
}

impl StdioServerTransport {
    pub(crate) fn process_handle(&self) -> StdioServerProcessHandle {
        self.process.clone()
    }
}

impl StdioServerCommand {
    /// Build the stdio process parameters before choosing where the process
    /// runs.
    pub(super) fn new(
        program: OsString,
        args: Vec<OsString>,
        env: Option<HashMap<OsString, OsString>>,
        env_vars: Vec<McpServerEnvVar>,
        cwd: Option<String>,
        protocol_mode: McpProtocolMode,
    ) -> Self {
        Self {
            program,
            args,
            env,
            env_vars,
            cwd,
            protocol_mode,
        }
    }
}

// Local public implementation.

/// Starts MCP stdio servers as local child processes.
///
/// This is the existing behavior for local MCP servers: the orchestrator
/// process spawns the configured command and rmcp talks to the child's local
/// stdin/stdout pipes directly.
#[derive(Clone)]
pub struct LocalStdioServerLauncher {
    fallback_cwd: PathBuf,
}

impl LocalStdioServerLauncher {
    /// Creates a local stdio launcher.
    ///
    /// `fallback_cwd` is used when the MCP server config omits `cwd`, so
    /// relative commands resolve from the caller's runtime working directory.
    pub fn new(fallback_cwd: PathBuf) -> Self {
        Self { fallback_cwd }
    }
}

#[cfg(any(target_os = "macos", test))]
const BROWSER_NODE_REPL_BROKER: &str = r###"
import { spawn } from "node:child_process";
import { unlink } from "node:fs/promises";
import net from "node:net";

const nodeReplPath = __NODE_REPL_PATH__;
const socketPath = "/tmp/codex-browser-broker-" + process.pid + ".sock";
try { await unlink(socketPath); } catch {}
const env = { ...process.env };
const broker = spawn(nodeReplPath, ["--disable-sandbox"], {
  env,
  stdio: ["pipe", "pipe", "inherit"],
});
broker.on("error", error => {
  process.stderr.write("broker spawn failed: " + (error.stack || error) + "\n");
  process.exitCode = 1;
});

const bootstrapCode = [
  'var cp = await import("node:child_process");',
  'var net = await import("node:net");',
  'var fs = await import("node:fs/promises");',
  "var browserSocketPath = " + JSON.stringify(socketPath) + ";",
  'try { await fs.unlink(browserSocketPath); } catch {}',
  'var browserChild = cp.spawn(' + JSON.stringify(nodeReplPath) + ', ["--disable-sandbox"], {env: ' + JSON.stringify(env) + ', stdio: ["pipe", "pipe", "pipe"]});',
  'browserChild.stderr.on("data", d => console.error(d.toString()));',
  'var brokerServer = net.createServer(client => {',
  '  if (globalThis.brokerClient) { client.destroy(); return; }',
  '  globalThis.brokerClient = client;',
  '  client.pipe(browserChild.stdin);',
  '  browserChild.stdout.pipe(client);',
  '  client.on("close", () => { try { browserChild.kill("SIGTERM"); } catch {} });',
  '});',
  'await new Promise((resolve, reject) => { brokerServer.once("error", reject); brokerServer.listen(browserSocketPath, () => resolve()); });',
  'nodeRepl.write(JSON.stringify({ready: true, path: browserSocketPath}));',
].join(" ");
const send = value => broker.stdin.write(JSON.stringify(value) + "\n");
send({
  jsonrpc: "2.0",
  id: 0,
  method: "initialize",
  params: {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "codex-signed-node-repl-broker", version: "1.0" },
  },
});
send({ jsonrpc: "2.0", method: "notifications/initialized" });
send({
  jsonrpc: "2.0",
  id: 1,
  method: "tools/call",
  params: {
    name: "js",
    arguments: { title: "start signed browser child", code: bootstrapCode },
  },
});

let bootstrapBuffer = "";
let connected = false;
broker.stdout.on("data", chunk => {
  bootstrapBuffer += chunk.toString();
  while (true) {
    const newline = bootstrapBuffer.indexOf("\n");
    if (newline < 0) break;
    const line = bootstrapBuffer.slice(0, newline);
    bootstrapBuffer = bootstrapBuffer.slice(newline + 1);
    let message;
    try { message = JSON.parse(line); } catch { continue; }
    const text = message?.result?.content?.find?.(item => item.type === "text")?.text;
    if (!connected && message?.id === 1) {
      if (message.error || message.result?.isError) {
        process.stderr.write("browser broker bootstrap failed: " + (text || JSON.stringify(message)) + "\n");
        process.exit(1);
      }
      let ready;
      try { ready = JSON.parse(text); } catch {}
      if (ready?.ready && ready.path === socketPath) {
        connected = true;
        const socket = net.createConnection(socketPath, () => {
          process.stdin.pipe(socket);
          socket.pipe(process.stdout);
          process.stdin.resume();
        });
        socket.on("error", error => {
          process.stderr.write("browser broker socket failed: " + (error.stack || error) + "\n");
          process.exitCode = 1;
        });
        socket.on("close", async () => {
          try { broker.kill("SIGTERM"); } catch {}
          try { await unlink(socketPath); } catch {}
          process.exit();
        });
      }
    }
  }
});
process.on("SIGTERM", () => { try { broker.kill("SIGTERM"); } catch {} });
process.on("SIGINT", () => { try { broker.kill("SIGINT"); } catch {} });
"###;

#[cfg(any(target_os = "macos", test))]
fn is_chatgpt_browser_node_repl(path: &Path) -> bool {
    let Some(bin_dir) = path.parent() else {
        return false;
    };
    let Some(cua_node_dir) = bin_dir.parent() else {
        return false;
    };
    let Some(resources_dir) = cua_node_dir.parent() else {
        return false;
    };
    let Some(contents_dir) = resources_dir.parent() else {
        return false;
    };
    let Some(app_dir) = contents_dir.parent() else {
        return false;
    };

    path.file_name().and_then(|name| name.to_str()) == Some("node_repl")
        && bin_dir.file_name().and_then(|name| name.to_str()) == Some("bin")
        && cua_node_dir.file_name().and_then(|name| name.to_str()) == Some("cua_node")
        && resources_dir.file_name().and_then(|name| name.to_str()) == Some("Resources")
        && contents_dir.file_name().and_then(|name| name.to_str()) == Some("Contents")
        && app_dir.extension().and_then(|extension| extension.to_str()) == Some("app")
        && app_dir.parent() == Some(Path::new("/Applications"))
}

#[cfg(any(target_os = "macos", test))]
fn browser_node_repl_broker_script(node_repl_path: &Path) -> String {
    let serialized_path =
        serde_json::Value::String(node_repl_path.to_string_lossy().into_owned()).to_string();
    BROWSER_NODE_REPL_BROKER.replace("__NODE_REPL_PATH__", &serialized_path)
}

#[cfg(any(target_os = "macos", test))]
fn browser_node_repl_broker_command(
    resolved_program: &Path,
    is_file: impl FnOnce(&Path) -> bool,
) -> Option<(PathBuf, Vec<OsString>)> {
    if !is_chatgpt_browser_node_repl(resolved_program) {
        return None;
    }
    let bin_dir = resolved_program.parent()?;
    let node_path = bin_dir.join("node");
    if !is_file(&node_path) {
        return None;
    }
    let broker_script = browser_node_repl_broker_script(resolved_program);
    Some((
        node_path,
        vec![
            OsString::from("--input-type=module"),
            OsString::from("-e"),
            OsString::from(broker_script),
        ],
    ))
}

#[cfg(target_os = "macos")]
fn maybe_wrap_browser_node_repl(
    resolved_program: PathBuf,
    args: Vec<OsString>,
) -> (PathBuf, Vec<OsString>) {
    browser_node_repl_broker_command(&resolved_program, Path::is_file)
        .unwrap_or((resolved_program, args))
}

#[cfg(not(target_os = "macos"))]
fn maybe_wrap_browser_node_repl(
    resolved_program: PathBuf,
    args: Vec<OsString>,
) -> (PathBuf, Vec<OsString>) {
    (resolved_program, args)
}
impl StdioServerLauncher for LocalStdioServerLauncher {
    fn launch(
        &self,
        command: StdioServerCommand,
    ) -> BoxFuture<'static, io::Result<StdioServerTransport>> {
        let fallback_cwd = self.fallback_cwd.clone();
        async move {
            // Keep synchronous program resolution and process creation from blocking the
            // caller's startup deadline.
            tokio::task::spawn_blocking(move || Self::launch_server(command, fallback_cwd))
                .await
                .map_err(io::Error::other)?
        }
        .boxed()
    }
}

// Local private implementation.

#[cfg(unix)]
const PROCESS_GROUP_TERM_GRACE_PERIOD: Duration = Duration::from_secs(2);

#[cfg(unix)]
struct LocalProcessTerminator {
    process_group_id: u32,
}

#[cfg(windows)]
struct LocalProcessTerminator {
    pid: u32,
}

#[cfg(not(any(unix, windows)))]
struct LocalProcessTerminator;

#[derive(Clone)]
pub(crate) struct StdioServerProcessHandle {
    inner: Arc<StdioServerProcessHandleInner>,
}

struct StdioServerProcessHandleInner {
    program_name: String,
    kind: StdioServerProcessKind,
    terminated: AtomicBool,
}

enum StdioServerProcessKind {
    Local(Option<LocalProcessTerminator>),
    Executor(Arc<dyn ExecProcess>),
}

mod private {
    pub trait Sealed {}
}

impl private::Sealed for LocalStdioServerLauncher {}

impl LocalStdioServerLauncher {
    fn launch_server(
        command: StdioServerCommand,
        fallback_cwd: PathBuf,
    ) -> io::Result<StdioServerTransport> {
        let StdioServerCommand {
            program,
            args,
            env,
            env_vars,
            cwd,
            protocol_mode,
        } = command;
        let program_name = program.to_string_lossy().into_owned();
        let envs = create_env_for_mcp_server(env, &env_vars).map_err(io::Error::other)?;
        let cwd = cwd.map(PathBuf::from).unwrap_or(fallback_cwd);
        let resolved_program =
            program_resolver::resolve(program, &envs, &cwd).map_err(io::Error::other)?;
        let (resolved_program, args) = maybe_wrap_browser_node_repl(resolved_program.into(), args);

        let mut command = Command::new(resolved_program);
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .current_dir(cwd)
            .env_clear()
            .envs(envs)
            .args(args);
        #[cfg(unix)]
        command.process_group(0);

        let (transport, stderr, process_id) = match protocol_mode {
            McpProtocolMode::Legacy => {
                let (transport, stderr) = TokioChildProcess::builder(command)
                    .stderr(Stdio::piped())
                    .spawn()?;
                let process_id = transport.id();
                (
                    StdioServerTransportInner::LocalLegacy(transport),
                    stderr,
                    process_id,
                )
            }
            McpProtocolMode::V20260728 => {
                let (transport, stderr) =
                    LocalStdioTransport::spawn(command, program_name.clone())?;
                let process_id = transport.id();
                (
                    StdioServerTransportInner::LocalModern(transport),
                    stderr,
                    process_id,
                )
            }
        };
        let process = StdioServerProcessHandle::local(
            program_name.clone(),
            process_id.map(LocalProcessTerminator::new),
        );

        if let Some(stderr) = stderr {
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                loop {
                    match reader.next_line().await {
                        Ok(Some(line)) => {
                            info!("MCP server stderr ({program_name}): {line}");
                        }
                        Ok(None) => break,
                        Err(error) => {
                            warn!("Failed to read MCP server stderr ({program_name}): {error}");
                            break;
                        }
                    }
                }
            });
        }

        Ok(StdioServerTransport {
            inner: transport,
            process,
        })
    }
}

impl LocalProcessTerminator {
    fn new(process_group_id: u32) -> Self {
        #[cfg(unix)]
        {
            Self { process_group_id }
        }
        #[cfg(windows)]
        {
            Self {
                pid: process_group_id,
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = process_group_id;
            Self
        }
    }

    #[cfg(unix)]
    fn terminate(&self) {
        let process_group_id = self.process_group_id;
        let should_escalate = match terminate_process_group(process_group_id) {
            Ok(exists) => exists,
            Err(error) => {
                warn!("Failed to terminate MCP process group {process_group_id}: {error}");
                false
            }
        };
        if should_escalate {
            spawn(move || {
                sleep(PROCESS_GROUP_TERM_GRACE_PERIOD);
                if let Err(error) = kill_process_group(process_group_id) {
                    warn!("Failed to kill MCP process group {process_group_id}: {error}");
                }
            });
        }
    }

    #[cfg(windows)]
    fn terminate(&self) {
        let _ = std::process::Command::new("taskkill")
            .arg("/PID")
            .arg(self.pid.to_string())
            .arg("/T")
            .arg("/F")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(not(any(unix, windows)))]
    fn terminate(&self) {}
}

impl StdioServerProcessHandle {
    fn local(program_name: String, terminator: Option<LocalProcessTerminator>) -> Self {
        Self {
            inner: Arc::new(StdioServerProcessHandleInner {
                program_name,
                kind: StdioServerProcessKind::Local(terminator),
                terminated: AtomicBool::new(false),
            }),
        }
    }

    pub(crate) fn executor(program_name: String, process: Arc<dyn ExecProcess>) -> Self {
        Self {
            inner: Arc::new(StdioServerProcessHandleInner {
                program_name,
                kind: StdioServerProcessKind::Executor(process),
                terminated: AtomicBool::new(false),
            }),
        }
    }

    pub(crate) async fn terminate(&self) -> io::Result<()> {
        if self.inner.terminated.swap(true, Ordering::AcqRel) {
            return Ok(());
        }

        match &self.inner.kind {
            StdioServerProcessKind::Local(Some(terminator)) => {
                terminator.terminate();
                Ok(())
            }
            StdioServerProcessKind::Local(None) => Ok(()),
            StdioServerProcessKind::Executor(process) => match process.terminate().await {
                Ok(()) => Ok(()),
                Err(error) => {
                    self.inner.terminated.store(false, Ordering::Release);
                    Err(io::Error::other(error))
                }
            },
        }
    }
}

impl Drop for StdioServerProcessHandleInner {
    fn drop(&mut self) {
        if self.terminated.swap(true, Ordering::AcqRel) {
            return;
        }

        match &self.kind {
            StdioServerProcessKind::Local(Some(terminator)) => {
                terminator.terminate();
            }
            StdioServerProcessKind::Local(None) => {}
            StdioServerProcessKind::Executor(process) => {
                let process = Arc::clone(process);
                let program_name = self.program_name.clone();
                let Ok(handle) = tokio::runtime::Handle::try_current() else {
                    warn!(
                        "Could not schedule remote MCP server process termination on drop ({}): no Tokio runtime is available",
                        self.program_name
                    );
                    return;
                };

                std::mem::drop(handle.spawn(async move {
                    if let Err(error) = process.terminate().await {
                        warn!(
                            "Failed to terminate remote MCP server process on drop ({program_name}): {error}"
                        );
                    }
                }));
            }
        }
    }
}

// Remote public implementation.

/// Starts MCP stdio servers through the executor process API.
///
/// MCP framing still runs in the orchestrator. The executor only owns the
/// child process and transports raw stdin/stdout/stderr bytes, so it does not
/// need to know about MCP methods such as `initialize` or `tools/list`.
#[derive(Clone)]
pub struct ExecutorStdioServerLauncher {
    exec_backend: Arc<dyn ExecBackend>,
}

impl ExecutorStdioServerLauncher {
    /// Creates a stdio server launcher backed by the executor process API.
    pub fn new(exec_backend: Arc<dyn ExecBackend>) -> Self {
        Self { exec_backend }
    }
}

impl StdioServerLauncher for ExecutorStdioServerLauncher {
    fn launch(
        &self,
        command: StdioServerCommand,
    ) -> BoxFuture<'static, io::Result<StdioServerTransport>> {
        let exec_backend = Arc::clone(&self.exec_backend);
        async move { Self::launch_server(command, exec_backend).await }.boxed()
    }
}

// Remote private implementation.

impl private::Sealed for ExecutorStdioServerLauncher {}

impl ExecutorStdioServerLauncher {
    async fn launch_server(
        command: StdioServerCommand,
        exec_backend: Arc<dyn ExecBackend>,
    ) -> io::Result<StdioServerTransport> {
        let StdioServerCommand {
            program,
            args,
            env,
            env_vars,
            cwd,
            protocol_mode: _,
        } = command;
        let Some(cwd) = cwd else {
            return Err(io::Error::other(
                "executor stdio server requires an explicit cwd",
            ));
        };
        let cwd: PathUri = LegacyAppPathString::from_path(Path::new(&cwd))
            .try_into()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        let program_name = program.to_string_lossy().into_owned();
        let envs = create_env_overlay_for_remote_mcp_server(env, &env_vars);
        let remote_env_vars = remote_mcp_env_var_names(&env_vars);
        // The executor protocol carries argv/env as UTF-8 strings. Local stdio can
        // accept arbitrary OsString values because it calls the OS directly; remote
        // stdio must reject non-Unicode command, argument, or environment data
        // before sending an executor request.
        let argv = Self::process_api_argv(&program, &args).map_err(io::Error::other)?;
        let env = Self::process_api_env(envs).map_err(io::Error::other)?;
        let process_id = ExecutorProcessTransport::next_process_id();
        // Start the MCP server process on the executor with raw pipes. `tty=false`
        // keeps stdout as a clean protocol stream, while `pipe_stdin=true` lets
        // rmcp write JSON-RPC requests after the process starts.
        let started = exec_backend
            .start(ExecParams {
                process_id,
                argv,
                cwd,
                env_policy: Some(Self::remote_env_policy(&remote_env_vars)),
                env,
                tty: false,
                pipe_stdin: true,
                arg0: None,
                sandbox: None,
                enforce_managed_network: false,
                managed_network: None,
                network_proxy: None,
            })
            .await
            .map_err(io::Error::other)?;

        let process =
            StdioServerProcessHandle::executor(program_name.clone(), Arc::clone(&started.process));
        Ok(StdioServerTransport {
            inner: StdioServerTransportInner::Executor(ExecutorProcessTransport::new(
                started.process,
                program_name,
            )),
            process,
        })
    }

    fn process_api_argv(program: &OsString, args: &[OsString]) -> Result<Vec<String>> {
        let mut argv = Vec::with_capacity(args.len() + 1);
        argv.push(Self::os_string_to_process_api_string(
            program.clone(),
            "command",
        )?);
        for arg in args {
            argv.push(Self::os_string_to_process_api_string(
                arg.clone(),
                "argument",
            )?);
        }
        Ok(argv)
    }

    fn process_api_env(env: HashMap<OsString, OsString>) -> Result<HashMap<String, String>> {
        env.into_iter()
            .map(|(key, value)| {
                Ok((
                    Self::os_string_to_process_api_string(key, "environment variable name")?,
                    Self::os_string_to_process_api_string(value, "environment variable value")?,
                ))
            })
            .collect()
    }

    fn os_string_to_process_api_string(value: OsString, label: &str) -> Result<String> {
        value
            .into_string()
            .map_err(|_| anyhow!("{label} must be valid Unicode for remote MCP stdio"))
    }

    fn remote_env_policy(remote_env_vars: &[String]) -> ExecEnvPolicy {
        let include_only = if remote_env_vars.is_empty() {
            Vec::new()
        } else {
            // `source = "remote"` means the value is read from the executor's
            // environment, not copied from Codex. Start from `All` only so the
            // named remote variable is available to the filter below; the
            // effective child env is still limited by `include_only`.
            crate::utils::DEFAULT_ENV_VARS
                .iter()
                .map(|name| (*name).to_string())
                .chain(remote_env_vars.iter().cloned())
                .collect()
        };
        ExecEnvPolicy {
            inherit: if remote_env_vars.is_empty() {
                ShellEnvironmentPolicyInherit::Core
            } else {
                ShellEnvironmentPolicyInherit::All
            },
            ignore_default_excludes: true,
            exclude: Vec::new(),
            r#set: HashMap::new(),
            include_only,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::config_types::EnvironmentVariablePattern;
    use codex_protocol::config_types::ShellEnvironmentPolicy;
    use codex_protocol::shell_environment;

    #[test]
    fn browser_node_repl_broker_requires_chatgpt_cua_node_layout() {
        assert_eq!(
            [
                is_chatgpt_browser_node_repl(Path::new(
                    "/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node_repl",
                )),
                is_chatgpt_browser_node_repl(Path::new(
                    "/Applications/ChatGPT Canary.app/Contents/Resources/cua_node/bin/node_repl",
                )),
                is_chatgpt_browser_node_repl(Path::new(
                    "/tmp/ChatGPT.app/Contents/Resources/cua_node/bin/node_repl",
                )),
                is_chatgpt_browser_node_repl(Path::new("/tmp/node_repl")),
                is_chatgpt_browser_node_repl(Path::new("/tmp/other/bin/node_repl")),
            ],
            [true, true, false, false, false]
        );
    }

    #[test]
    fn browser_node_repl_broker_requires_signed_node_sibling() {
        let node_repl_path =
            Path::new("/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node_repl");
        assert_eq!(
            [
                browser_node_repl_broker_command(node_repl_path, |_| false).is_some(),
                browser_node_repl_broker_command(node_repl_path, |_| true).is_some(),
            ],
            [false, true]
        );
    }

    #[test]
    fn browser_node_repl_broker_serializes_quoted_path_once() {
        let script = browser_node_repl_broker_script(Path::new(
            "/Applications/ChatGPT \"Canary\".app/Contents/Resources/cua_node/bin/node_repl",
        ));

        assert_eq!(
            (
                script.contains("__NODE_REPL_PATH__"),
                script.contains(
                    r#"const nodeReplPath = "/Applications/ChatGPT \"Canary\".app/Contents/Resources/cua_node/bin/node_repl";"#,
                ),
                script.contains("const nodeReplPath = \"\""),
            ),
            (false, true, false)
        );
    }

    #[test]
    fn remote_env_policy_uses_core_env_without_remote_source_vars() {
        let policy = ExecutorStdioServerLauncher::remote_env_policy(&[]);

        assert_eq!(policy.inherit, ShellEnvironmentPolicyInherit::Core);
        assert!(policy.include_only.is_empty());
    }

    #[test]
    fn remote_env_policy_includes_remote_source_vars_without_full_env() {
        let policy = ExecutorStdioServerLauncher::remote_env_policy(&["REMOTE_TOKEN".to_string()]);

        assert_eq!(policy.inherit, ShellEnvironmentPolicyInherit::All);
        assert!(
            policy.include_only.contains(&"REMOTE_TOKEN".to_string()),
            "remote source var should be included in executor env policy"
        );
        assert!(
            policy
                .include_only
                .contains(&crate::utils::DEFAULT_ENV_VARS[0].to_string()),
            "remote default env vars should remain available"
        );
    }

    #[test]
    fn remote_env_policy_effectively_filters_unrequested_vars() {
        let exec_policy =
            ExecutorStdioServerLauncher::remote_env_policy(&["REMOTE_TOKEN".to_string()]);
        let policy = ShellEnvironmentPolicy {
            inherit: exec_policy.inherit,
            ignore_default_excludes: exec_policy.ignore_default_excludes,
            exclude: exec_policy
                .exclude
                .iter()
                .map(|pattern| EnvironmentVariablePattern::new_case_insensitive(pattern))
                .collect(),
            r#set: exec_policy.r#set,
            include_only: exec_policy
                .include_only
                .iter()
                .map(|pattern| EnvironmentVariablePattern::new_case_insensitive(pattern))
                .collect(),
            use_profile: false,
        };

        let env = shell_environment::create_env_from_vars(
            [
                ("PATH".to_string(), "/remote/bin".to_string()),
                ("REMOTE_TOKEN".to_string(), "remote-secret".to_string()),
                (
                    "UNREQUESTED_SECRET".to_string(),
                    "must-not-pass".to_string(),
                ),
            ],
            &policy,
            /*thread_id*/ None,
        );

        assert_eq!(env.get("PATH").map(String::as_str), Some("/remote/bin"));
        assert_eq!(
            env.get("REMOTE_TOKEN").map(String::as_str),
            Some("remote-secret")
        );
        assert!(!env.contains_key("UNREQUESTED_SECRET"));
    }
}
