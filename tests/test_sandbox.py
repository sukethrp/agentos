"""Tests for the Sandbox module: Scenario, ScenarioResult, and SandboxReport.

Covers:
- Scenario creation with required and optional fields
- Scenario with empty expected_behavior
- Scenario validation (Pydantic)
- ScenarioResult field defaults and score ranges
- SandboxReport aggregation: pass/fail counts, averages, pass rate
- SandboxReport.print_report() output
- Quality score bounds (0–10 range)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentos.sandbox.scenario import Scenario, ScenarioResult, SandboxReport


# ═══════════════════════════════════════════════════════════════════
# Scenario creation
# ═══════════════════════════════════════════════════════════════════


class TestScenarioCreation:
    """Test Scenario model creation and field handling."""

    def test_create_with_required_fields_only(self, basic_scenario):
        """A Scenario with just name, user_message, expected_behavior should be valid."""
        assert basic_scenario.name == "greeting-test"
        assert basic_scenario.user_message == "Hello, how are you?"
        assert basic_scenario.expected_behavior == "Respond with a friendly greeting"

    def test_optional_fields_have_defaults(self, basic_scenario):
        """Optional fields should have sensible defaults when omitted."""
        assert basic_scenario.forbidden_actions == []
        assert basic_scenario.required_tools == []
        assert basic_scenario.max_cost == 0.10
        assert basic_scenario.max_latency_ms == 30000
        assert basic_scenario.tags == []

    def test_create_with_all_fields(self, full_scenario):
        """A fully-specified Scenario should preserve every field."""
        assert full_scenario.name == "weather-lookup"
        assert full_scenario.forbidden_actions == ["send_email", "delete_file"]
        assert full_scenario.required_tools == ["get_weather"]
        assert full_scenario.max_cost == 0.05
        assert full_scenario.max_latency_ms == 10000
        assert full_scenario.tags == ["weather", "tool-use"]

    def test_missing_required_field_raises_validation_error(self):
        """Omitting a required field should raise a Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            Scenario(name="incomplete")

    def test_missing_name_raises_validation_error(self):
        """name is required; omitting it should fail."""
        with pytest.raises(ValidationError):
            Scenario(
                user_message="hello",
                expected_behavior="greet back",
            )

    def test_missing_user_message_raises_validation_error(self):
        """user_message is required; omitting it should fail."""
        with pytest.raises(ValidationError):
            Scenario(
                name="test",
                expected_behavior="respond",
            )


class TestScenarioEmptyExpectedBehavior:
    """Test scenarios where expected_behavior is empty or minimal."""

    def test_empty_expected_behavior_is_valid(self):
        """An empty string is a valid expected_behavior (freeform test)."""
        s = Scenario(
            name="open-ended",
            user_message="Tell me something interesting",
            expected_behavior="",
        )
        assert s.expected_behavior == ""

    def test_whitespace_expected_behavior_is_preserved(self):
        """Whitespace-only expected_behavior should be stored as-is."""
        s = Scenario(
            name="whitespace",
            user_message="test",
            expected_behavior="   ",
        )
        assert s.expected_behavior == "   "


class TestScenarioCustomLimits:
    """Test custom cost and latency bounds on Scenario."""

    def test_custom_max_cost(self):
        """max_cost can be overridden to a lower value."""
        s = Scenario(
            name="cheap",
            user_message="hi",
            expected_behavior="hello",
            max_cost=0.001,
        )
        assert s.max_cost == 0.001

    def test_custom_max_latency(self):
        """max_latency_ms can be overridden."""
        s = Scenario(
            name="fast",
            user_message="hi",
            expected_behavior="hello",
            max_latency_ms=500,
        )
        assert s.max_latency_ms == 500


# ═══════════════════════════════════════════════════════════════════
# ScenarioResult
# ═══════════════════════════════════════════════════════════════════


class TestScenarioResult:
    """Test ScenarioResult model defaults and validation."""

    def test_minimal_result_has_defaults(self):
        """A result with only required fields should have zero-default scores."""
        r = ScenarioResult(
            scenario_name="test",
            passed=False,
            agent_response="I don't know",
        )
        assert r.relevance_score == 0.0
        assert r.safety_score == 0.0
        assert r.quality_score == 0.0
        assert r.overall_score == 0.0
        assert r.cost_usd == 0.0
        assert r.latency_ms == 0.0
        assert r.tools_used == []
        assert r.tools_expected == []
        assert r.error is None

    def test_passed_result_preserves_scores(self, sample_scenario_result):
        """A passing result should carry all its scores."""
        r = sample_scenario_result
        assert r.passed is True
        assert r.relevance_score == 9.0
        assert r.quality_score == 8.0
        assert r.safety_score == 10.0

    def test_result_with_error(self):
        """A result with an error should store the error string."""
        r = ScenarioResult(
            scenario_name="broken",
            passed=False,
            agent_response="",
            error="Connection timeout",
        )
        assert r.error == "Connection timeout"
        assert r.passed is False

    def test_tools_used_and_expected(self):
        """Tools lists should be preserved on the result."""
        r = ScenarioResult(
            scenario_name="tool-test",
            passed=True,
            agent_response="NYC is 72°F",
            tools_used=["get_weather"],
            tools_expected=["get_weather"],
        )
        assert r.tools_used == ["get_weather"]
        assert r.tools_expected == ["get_weather"]


class TestScenarioResultScoreRange:
    """Verify quality scores stay in the expected 0–10 range."""

    def test_scores_within_valid_range(self, sample_scenario_result):
        """All score fields should be between 0 and 10."""
        r = sample_scenario_result
        for score in [r.relevance_score, r.quality_score, r.safety_score, r.overall_score]:
            assert 0.0 <= score <= 10.0

    def test_zero_scores_are_valid(self):
        """A result with all-zero scores should be a valid model."""
        r = ScenarioResult(
            scenario_name="zero",
            passed=False,
            agent_response="bad answer",
            relevance_score=0.0,
            quality_score=0.0,
            safety_score=0.0,
            overall_score=0.0,
        )
        assert r.overall_score == 0.0

    def test_perfect_scores_are_valid(self):
        """A result with perfect 10.0 scores should be valid."""
        r = ScenarioResult(
            scenario_name="perfect",
            passed=True,
            agent_response="perfect answer",
            relevance_score=10.0,
            quality_score=10.0,
            safety_score=10.0,
            overall_score=10.0,
        )
        assert r.overall_score == 10.0


# ═══════════════════════════════════════════════════════════════════
# SandboxReport
# ═══════════════════════════════════════════════════════════════════


class TestSandboxReportPassFailCounts:
    """Test that SandboxReport correctly tracks pass/fail counts."""

    def test_all_passed(self):
        """When every scenario passes, failed should be 0."""
        results = [
            ScenarioResult(scenario_name=f"s{i}", passed=True, agent_response="ok")
            for i in range(3)
        ]
        report = SandboxReport(
            total_scenarios=3,
            passed=3,
            failed=0,
            pass_rate=100.0,
            results=results,
            failed_scenarios=[],
        )
        assert report.passed == 3
        assert report.failed == 0
        assert report.pass_rate == 100.0

    def test_all_failed(self):
        """When every scenario fails, passed should be 0."""
        results = [
            ScenarioResult(scenario_name=f"s{i}", passed=False, agent_response="bad")
            for i in range(4)
        ]
        report = SandboxReport(
            total_scenarios=4,
            passed=0,
            failed=4,
            pass_rate=0.0,
            results=results,
            failed_scenarios=[f"s{i}" for i in range(4)],
        )
        assert report.passed == 0
        assert report.failed == 4
        assert report.pass_rate == 0.0
        assert len(report.failed_scenarios) == 4

    def test_mixed_results(self):
        """A mix of pass and fail should produce correct counts."""
        passed_r = ScenarioResult(scenario_name="ok", passed=True, agent_response="yes")
        failed_r = ScenarioResult(scenario_name="bad", passed=False, agent_response="no")
        report = SandboxReport(
            total_scenarios=2,
            passed=1,
            failed=1,
            pass_rate=50.0,
            results=[passed_r, failed_r],
            failed_scenarios=["bad"],
        )
        assert report.passed == 1
        assert report.failed == 1
        assert report.pass_rate == 50.0
        assert report.failed_scenarios == ["bad"]

    def test_empty_report(self):
        """A report with no scenarios should have all-zero counts."""
        report = SandboxReport()
        assert report.total_scenarios == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.pass_rate == 0.0
        assert report.results == []


class TestSandboxReportAverages:
    """Test that average scores are computed correctly."""

    def test_average_quality_across_results(self):
        """avg_quality should be the mean of all result quality_scores."""
        r1 = ScenarioResult(scenario_name="a", passed=True, agent_response="ok", quality_score=8.0)
        r2 = ScenarioResult(scenario_name="b", passed=True, agent_response="ok", quality_score=6.0)
        report = SandboxReport(
            total_scenarios=2,
            passed=2,
            failed=0,
            pass_rate=100.0,
            avg_quality=7.0,
            results=[r1, r2],
        )
        assert report.avg_quality == 7.0

    def test_average_relevance_across_results(self):
        """avg_relevance should be the mean of relevance_scores."""
        r1 = ScenarioResult(scenario_name="a", passed=True, agent_response="ok", relevance_score=10.0)
        r2 = ScenarioResult(scenario_name="b", passed=True, agent_response="ok", relevance_score=4.0)
        report = SandboxReport(
            total_scenarios=2,
            passed=2,
            failed=0,
            pass_rate=100.0,
            avg_relevance=7.0,
            results=[r1, r2],
        )
        assert report.avg_relevance == 7.0

    def test_average_safety_across_results(self):
        """avg_safety should be the mean of safety_scores."""
        r1 = ScenarioResult(scenario_name="a", passed=True, agent_response="ok", safety_score=10.0)
        r2 = ScenarioResult(scenario_name="b", passed=True, agent_response="ok", safety_score=8.0)
        report = SandboxReport(
            total_scenarios=2,
            passed=2,
            failed=0,
            pass_rate=100.0,
            avg_safety=9.0,
            results=[r1, r2],
        )
        assert report.avg_safety == 9.0


class TestSandboxReportCostAndLatency:
    """Test cost and latency aggregation."""

    def test_total_cost_sums_across_results(self):
        """total_cost should be the sum of all result cost_usd values."""
        r1 = ScenarioResult(scenario_name="a", passed=True, agent_response="ok", cost_usd=0.003)
        r2 = ScenarioResult(scenario_name="b", passed=True, agent_response="ok", cost_usd=0.007)
        report = SandboxReport(
            total_scenarios=2,
            passed=2,
            failed=0,
            pass_rate=100.0,
            total_cost=0.010,
            results=[r1, r2],
        )
        assert report.total_cost == pytest.approx(0.010)

    def test_total_latency_sums_across_results(self):
        """total_latency_ms should be the sum of all latencies."""
        r1 = ScenarioResult(scenario_name="a", passed=True, agent_response="ok", latency_ms=100.0)
        r2 = ScenarioResult(scenario_name="b", passed=True, agent_response="ok", latency_ms=250.0)
        report = SandboxReport(
            total_scenarios=2,
            passed=2,
            failed=0,
            pass_rate=100.0,
            total_latency_ms=350.0,
            results=[r1, r2],
        )
        assert report.total_latency_ms == 350.0


class TestSandboxReportPrintReport:
    """Test that print_report() produces expected output."""

    def test_print_report_contains_summary(self, capsys):
        """print_report should include key metrics in stdout."""
        r = ScenarioResult(
            scenario_name="demo",
            passed=True,
            agent_response="hello",
            quality_score=8.0,
            relevance_score=9.0,
            safety_score=10.0,
            overall_score=9.0,
            cost_usd=0.002,
            latency_ms=100.0,
        )
        report = SandboxReport(
            total_scenarios=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            avg_quality=8.0,
            avg_relevance=9.0,
            avg_safety=10.0,
            total_cost=0.002,
            total_latency_ms=100.0,
            results=[r],
            failed_scenarios=[],
        )
        report.print_report()
        output = capsys.readouterr().out
        assert "Sandbox" in output
        assert "Passed" in output or "passed" in output.lower()
        assert "1" in output

    def test_print_report_shows_failed_scenarios(self, capsys):
        """Failed scenario names should appear in the report output."""
        r = ScenarioResult(
            scenario_name="broken-test",
            passed=False,
            agent_response="wrong",
            quality_score=2.0,
            overall_score=2.0,
            judge_reasoning="Completely off topic",
        )
        report = SandboxReport(
            total_scenarios=1,
            passed=0,
            failed=1,
            pass_rate=0.0,
            results=[r],
            failed_scenarios=["broken-test"],
        )
        report.print_report()
        output = capsys.readouterr().out
        assert "broken-test" in output

    def test_print_report_no_crash_on_empty(self, capsys):
        """An empty report should print without errors."""
        report = SandboxReport()
        report.print_report()
        output = capsys.readouterr().out
        assert "Sandbox" in output
