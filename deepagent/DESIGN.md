# deepagent

A reusable runtime for driving an in-process coding agent through a bounded,
resumable turn loop. Extracted from icompleteu so a new application -- a
different job spec and state machine -- is low overhead: you write the state
machine, the prompts, and the verdict schemas; you import the turn engine, the
tools, the caching/accounting, and the robustness.

## The boundary

The split is *turn engine* (reusable) vs *pipeline* (application). A state
machine is inherently domain-specific and stays in the application; what
generalizes is "run one bounded agent turn to a typed verdict."

**deepagent (the runtime) owns:**
- `Harness` Protocol, `AgentResult`, `Disposition`, `HarnessRun` -- the turn
  contract.
- `Report` (base structured verdict: disposition/summary/reason) and the
  `report -> AgentResult` flattening.
- `run_agent_loop` + `LoopCaps` -- the reentry-to-terminal-disposition loop.
  Schema- and domain-agnostic; it only knows dispositions.
- `LangGraphHarness` -- the in-process `create_agent` turn, reusing airc-core's
  middleware (context budget, empty-response strip, retry, Anthropic caching),
  the Vertex growing-prefix cache, the call-budget governor, and the shared
  token ledger. Worktree-bound shell/read/edit tools (airc-tools).
- `MockHarness` -- the test double.
- `render_skill_index` -- turn a directory of frontmatter'd skill files into a
  sysprompt index (progressive disclosure; see Skills).

**The application owns:**
- Its state machine (stages, transitions, terminal handling).
- Its job spec.
- Its stage prompts and its per-stage `Report` subclasses (the verdict fields).
- Its system prompt: identity + conventions + the skill index it chose.
- Any domain glue (for icompleteu: worktree via `vt`, gerrit/CQ/Pinpoint forge).

The `Harness` Protocol abstracts the backend, which is why the loop and machine
did not change when the backend went from jetski-subprocess to in-process
langgraph. A future subprocess backend can live behind the same Protocol.

## The injected surface

`LangGraphHarness(common, *, system_prompt, schemas, coding_model_key,
coding_tool_groups, shell_timeout_s)`:
- `common: CommonConfig` -- model/mcp/tool_groups/caching/token_db, from
  `airc_core.load_common` (the shared suite config).
- `system_prompt: str` -- the app's composed base prompt (identity +
  conventions + skill index). MCP server instructions are appended by the
  harness.
- `schemas: Mapping[str, type[Report]]` -- agent-name -> verdict schema. The
  turn uses `schemas.get(agent, Report)`; the state machine names the agent per
  stage, so each stage gets its own verdict shape. Unmapped agents fall back to
  the bare `Report`.
- `coding_model_key`, `coding_tool_groups`, `shell_timeout_s` -- as today.

`run_agent_loop(harness, *, prompt_path, workdir, control_dir, caps, agent,
casefile)` is unchanged: it drives `harness.run_once` to a terminal disposition
and returns an `AgentResult`. The application's stage body writes the turn
prompt, calls this, reads `res.disposition` and `res.data.*`.

Nothing else about the runtime is application-aware. In particular the runtime
carries no domain constants (no stage names, no application identity, no verdict
fields).

## Skills

Progressive disclosure, not "everything in the sysprompt." A skill is a
markdown file with `name`/`description` frontmatter; the *index* (name + one
line + path) goes in the cached system prompt, the *body* is read on demand.

For a V8 application the store is a knowledge-base repo: it is already a v8-utils
MCP repo, and `repo_git_*` is already in the `read` tool group the agent gets, so
the loader exists with no new tooling. The body is read via
`repo_git_show(repo="kb", path="skills/...")`; the read lands in the conversation
tail, so the context budget middleware sheds it when stale -- a skill needed at
turn 2 does not tax turn 20. The index, being in the stable sysprompt, is cached
across turns.

Convention:
- Skills live under `<kb>/skills/` (distinct from investigation writeups, so
  the index stays a curated set, not the whole store).
- Each file starts with frontmatter: `name`, `description` (one line). The body
  is the playbook.
- `deepagent.render_skill_index(skill_dir)` reads the frontmatter and returns
  the index block; the application concatenates it into its system prompt at
  harness-build time (self-maintaining: add a skill file, the index updates, no
  code change). Grep is the fallback for what the index did not surface, not the
  primary path.

`render_skill_index` is generic (a directory of frontmatter'd files -> index
string); the *store choice* and the read path are the application's.

## Building a new application

1. Define the job spec and the state machine (stages; `advance` dispatch;
   terminal handling). Reuse `run_agent_loop` for the agent-driven stages.
2. Per agent-driven stage, define a `Report` subclass with the verdict fields
   the pipeline reads, and add it to the `schemas` map under the stage's agent
   name.
3. Compose the system prompt: identity + conventions +
   `render_skill_index(skill_dir)`.
4. Construct `LangGraphHarness(common, system_prompt=..., schemas=...)`, wire it
   into the loop caps, drive the machine.

The domain glue (worktrees, a forge, a bus) is yours; the turn engine, tools,
caching, token accounting, cancellation-kills-the-build robustness, and the
skill index are imported.
