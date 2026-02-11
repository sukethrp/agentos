"""Multi-Agent Teams — Agents that collaborate, delegate, and debate.

Unlike CrewAI, AgentOS teams have:
- Built-in governance (budget shared or per-agent)
- Full audit trail of inter-agent communication
- Simulation sandbox testing for the entire team
- Configurable orchestration patterns (sequential, parallel, debate, hierarchy)

Usage:
    team = AgentTeam(
        name="research-team",
        agents=[researcher, analyst, writer],
        strategy="sequential",
    )
    result = team.run("Write a report on AI agent market trends")
"""

from __future__ import annotations
import time
from typing import Any
from pydantic import BaseModel, Field
from agentos.core.agent import Agent
from agentos.core.types import Message, Role, AgentEvent


class TeamMessage(BaseModel):
    """A message passed between agents in a team."""
    from_agent: str
    to_agent: str
    content: str
    timestamp: float = Field(default_factory=time.time)
    message_type: str = "task"  # task, result, feedback, delegate


class TeamResult(BaseModel):
    """Result from a team execution."""
    final_output: str
    steps: list[dict] = Field(default_factory=list)
    messages: list[TeamMessage] = Field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    agents_used: list[str] = Field(default_factory=list)


class AgentTeam:
    """A team of agents that work together.

    Strategies:
    - "sequential": Each agent works in order, passing output to next
    - "parallel": All agents work on the same task, best result picked
    - "debate": Agents propose solutions and critique each other
    - "hierarchy": Manager agent delegates to specialist agents
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent],
        strategy: str = "sequential",
        manager: Agent | None = None,
        max_rounds: int = 3,
    ):
        self.name = name
        self.agents = agents
        self._agent_map = {a.config.name: a for a in agents}
        self.strategy = strategy
        self.manager = manager
        self.max_rounds = max_rounds
        self.messages: list[TeamMessage] = []
        self.events: list[AgentEvent] = []

    def run(self, task: str) -> TeamResult:
        """Run the team on a task using the configured strategy."""
        print(f"\n{'='*60}")
        print(f"👥 Team [{self.name}] — Strategy: {self.strategy}")
        print(f"   Agents: {', '.join(a.config.name for a in self.agents)}")
        print(f"   Task: {task}")
        print(f"{'='*60}")

        start = time.time()

        if self.strategy == "sequential":
            result = self._run_sequential(task)
        elif self.strategy == "parallel":
            result = self._run_parallel(task)
        elif self.strategy == "debate":
            result = self._run_debate(task)
        elif self.strategy == "hierarchy":
            result = self._run_hierarchy(task)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        result.total_time_ms = round((time.time() - start) * 1000, 2)
        result.agents_used = [a.config.name for a in self.agents]
        result.messages = self.messages.copy()

        self._print_summary(result)
        return result

    def _run_sequential(self, task: str) -> TeamResult:
        """Each agent works in sequence, building on previous output."""
        steps = []
        current_input = task
        total_cost = 0.0
        total_tokens = 0

        for i, agent in enumerate(self.agents):
            print(f"\n📌 Step {i+1}/{len(self.agents)}: {agent.config.name}")
            print(f"   Input: {current_input[:100]}...")

            # Build prompt that includes context from previous agents
            if i == 0:
                prompt = current_input
            else:
                prompt = f"Previous work from the team:\n{current_input}\n\nYour task: Continue and improve this work. Add your expertise."

            msg = agent.run(prompt)
            output = msg.content or ""

            # Track costs
            cost = sum(e.cost_usd for e in agent.events)
            tokens = sum(e.tokens_used for e in agent.events)
            total_cost += cost
            total_tokens += tokens

            # Log inter-agent message
            next_agent = self.agents[i+1].config.name if i+1 < len(self.agents) else "output"
            self.messages.append(TeamMessage(
                from_agent=agent.config.name,
                to_agent=next_agent,
                content=output[:500],
                message_type="result",
            ))

            steps.append({
                "agent": agent.config.name,
                "input": current_input[:200],
                "output": output[:500],
                "cost": cost,
                "tokens": tokens,
            })

            current_input = output

        return TeamResult(
            final_output=current_input,
            steps=steps,
            total_cost=round(total_cost, 6),
            total_tokens=total_tokens,
        )

    def _run_parallel(self, task: str) -> TeamResult:
        """All agents work on the same task independently. Best result wins."""
        steps = []
        results = []
        total_cost = 0.0
        total_tokens = 0

        for i, agent in enumerate(self.agents):
            print(f"\n📌 Agent {i+1}/{len(self.agents)}: {agent.config.name} (parallel)")

            msg = agent.run(task)
            output = msg.content or ""

            cost = sum(e.cost_usd for e in agent.events)
            tokens = sum(e.tokens_used for e in agent.events)
            total_cost += cost
            total_tokens += tokens

            results.append({"agent": agent.config.name, "output": output, "cost": cost})
            steps.append({
                "agent": agent.config.name,
                "output": output[:500],
                "cost": cost,
                "tokens": tokens,
            })

        # Use the last agent (or manager) to pick the best result
        picker = self.manager or self.agents[-1]
        comparison = "\n\n".join([
            f"--- {r['agent']} ---\n{r['output']}" for r in results
        ])
        pick_prompt = f"Multiple agents worked on this task: '{task}'\n\nHere are their responses:\n{comparison}\n\nPick the BEST response and explain why. Return only the best response, improved if possible."

        print(f"\n📌 Picking best result using: {picker.config.name}")
        best_msg = picker.run(pick_prompt)
        total_cost += sum(e.cost_usd for e in picker.events)
        total_tokens += sum(e.tokens_used for e in picker.events)

        return TeamResult(
            final_output=best_msg.content or "",
            steps=steps,
            total_cost=round(total_cost, 6),
            total_tokens=total_tokens,
        )

    def _run_debate(self, task: str) -> TeamResult:
        """Agents propose solutions and critique each other for N rounds."""
        steps = []
        total_cost = 0.0
        total_tokens = 0
        proposals = {}

        # Round 1: Initial proposals
        print(f"\n🗣️ Round 1: Initial Proposals")
        for agent in self.agents:
            msg = agent.run(f"Propose your best solution for: {task}")
            proposals[agent.config.name] = msg.content or ""
            cost = sum(e.cost_usd for e in agent.events)
            tokens = sum(e.tokens_used for e in agent.events)
            total_cost += cost
            total_tokens += tokens

            steps.append({
                "round": 1,
                "agent": agent.config.name,
                "type": "proposal",
                "output": (msg.content or "")[:500],
            })

        # Rounds 2+: Critique and improve
        for round_num in range(2, self.max_rounds + 1):
            print(f"\n🗣️ Round {round_num}: Critique & Improve")
            new_proposals = {}

            for agent in self.agents:
                others = "\n\n".join([
                    f"--- {name} ---\n{prop}"
                    for name, prop in proposals.items()
                    if name != agent.config.name
                ])

                critique_prompt = f"Task: {task}\n\nYour previous proposal:\n{proposals[agent.config.name]}\n\nOther proposals:\n{others}\n\nCritique the other proposals and improve your own. Give your final improved answer."

                msg = agent.run(critique_prompt)
                new_proposals[agent.config.name] = msg.content or ""
                cost = sum(e.cost_usd for e in agent.events)
                tokens = sum(e.tokens_used for e in agent.events)
                total_cost += cost
                total_tokens += tokens

                self.messages.append(TeamMessage(
                    from_agent=agent.config.name,
                    to_agent="team",
                    content=(msg.content or "")[:300],
                    message_type="feedback",
                ))

                steps.append({
                    "round": round_num,
                    "agent": agent.config.name,
                    "type": "critique",
                    "output": (msg.content or "")[:500],
                })

            proposals = new_proposals

        # Final synthesis
        all_final = "\n\n".join([f"--- {n} ---\n{p}" for n, p in proposals.items()])
        synthesizer = self.manager or self.agents[0]
        print(f"\n📌 Final synthesis by: {synthesizer.config.name}")
        final = synthesizer.run(f"Synthesize these refined proposals into one final answer for: {task}\n\n{all_final}")
        total_cost += sum(e.cost_usd for e in synthesizer.events)
        total_tokens += sum(e.tokens_used for e in synthesizer.events)

        return TeamResult(
            final_output=final.content or "",
            steps=steps,
            total_cost=round(total_cost, 6),
            total_tokens=total_tokens,
        )

    def _run_hierarchy(self, task: str) -> TeamResult:
        """Manager delegates subtasks to specialist agents."""
        if not self.manager:
            self.manager = self.agents[0]

        steps = []
        total_cost = 0.0
        total_tokens = 0

        # Manager creates a plan
        agent_list = ", ".join([f"{a.config.name}" for a in self.agents if a != self.manager])
        plan_prompt = f"You are a manager. You have these team members: {agent_list}\n\nTask: {task}\n\nCreate a plan. For each step, specify which agent should do it and what their specific subtask is. Format: AGENT_NAME: subtask description (one per line)"

        print(f"\n📌 Manager [{self.manager.config.name}] creating plan...")
        plan_msg = self.manager.run(plan_prompt)
        plan = plan_msg.content or ""
        total_cost += sum(e.cost_usd for e in self.manager.events)
        total_tokens += sum(e.tokens_used for e in self.manager.events)

        steps.append({"agent": self.manager.config.name, "type": "plan", "output": plan[:500]})

        # Execute each subtask with the assigned agent
        subtask_results = []
        for line in plan.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            for agent in self.agents:
                if agent.config.name.lower() in line.lower() and agent != self.manager:
                    subtask = line.split(":", 1)[-1].strip()
                    print(f"\n📌 Delegating to {agent.config.name}: {subtask[:80]}...")

                    msg = agent.run(subtask)
                    result = msg.content or ""
                    cost = sum(e.cost_usd for e in agent.events)
                    tokens = sum(e.tokens_used for e in agent.events)
                    total_cost += cost
                    total_tokens += tokens

                    subtask_results.append(f"{agent.config.name}: {result}")

                    self.messages.append(TeamMessage(
                        from_agent=self.manager.config.name,
                        to_agent=agent.config.name,
                        content=subtask,
                        message_type="delegate",
                    ))

                    steps.append({
                        "agent": agent.config.name,
                        "type": "subtask",
                        "input": subtask[:200],
                        "output": result[:500],
                    })
                    break

        # Manager synthesizes results
        all_results = "\n\n".join(subtask_results)
        synthesis_prompt = f"Your team completed the subtasks. Here are their results:\n\n{all_results}\n\nSynthesize into a final comprehensive answer for: {task}"

        print(f"\n📌 Manager synthesizing final result...")
        final = self.manager.run(synthesis_prompt)
        total_cost += sum(e.cost_usd for e in self.manager.events)
        total_tokens += sum(e.tokens_used for e in self.manager.events)

        return TeamResult(
            final_output=final.content or "",
            steps=steps,
            total_cost=round(total_cost, 6),
            total_tokens=total_tokens,
        )

    def _print_summary(self, result: TeamResult):
        print(f"\n{'='*60}")
        print(f"👥 Team Run Summary: {self.name}")
        print(f"{'='*60}")
        print(f"   Strategy:     {self.strategy}")
        print(f"   Agents used:  {', '.join(result.agents_used)}")
        print(f"   Steps:        {len(result.steps)}")
        print(f"   Messages:     {len(result.messages)}")
        print(f"   Total cost:   ${result.total_cost:.4f}")
        print(f"   Total tokens: {result.total_tokens:,}")
        print(f"   Total time:   {result.total_time_ms:.0f}ms")
        print(f"{'='*60}")
        print(f"\n📝 Final Output:")
        print(result.final_output)
        print(f"{'='*60}")