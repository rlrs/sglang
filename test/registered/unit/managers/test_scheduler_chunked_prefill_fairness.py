"""Tests for bounded short-request interleaving during chunked prefill."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _request(*, prompt_len: int, beam: bool = False):
    return SimpleNamespace(
        beam_group=object() if beam else None,
        full_untruncated_fill_ids=list(range(prompt_len)),
        init_next_round_input=MagicMock(),
    )


def _scheduler(*, enabled: bool, active, waiting, bypass_due: bool = True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_fair_chunked_prefill = enabled
    scheduler._chunked_prefill_bypass_due = bypass_due
    scheduler._fair_chunked_prefill_decode_due = False
    scheduler.chunked_req = active
    scheduler.waiting_queue = waiting
    return scheduler


class TestChunkedPrefillFairness(unittest.TestCase):
    CHUNK_SIZE = 64

    def test_short_request_parks_active_chunk_for_one_turn(self):
        active = _request(prompt_len=4 * self.CHUNK_SIZE)
        short = _request(prompt_len=8)
        scheduler = _scheduler(enabled=True, active=active, waiting=[short])
        adder = MagicMock()

        batch_chunked_req, parked = scheduler._schedule_chunked_req_for_prefill(
            adder, self.CHUNK_SIZE
        )

        self.assertTrue(parked)
        self.assertIsNone(batch_chunked_req)
        self.assertIs(scheduler.chunked_req, active)
        self.assertFalse(scheduler._chunked_prefill_bypass_due)
        active.init_next_round_input.assert_not_called()
        adder.add_chunked_req.assert_not_called()

    def test_active_chunk_resumes_after_one_bypass(self):
        active = _request(prompt_len=4 * self.CHUNK_SIZE)
        short = _request(prompt_len=8)
        scheduler = _scheduler(
            enabled=True, active=active, waiting=[short], bypass_due=False
        )
        adder = MagicMock()
        adder.add_chunked_req.return_value = active

        batch_chunked_req, parked = scheduler._schedule_chunked_req_for_prefill(
            adder, self.CHUNK_SIZE
        )

        self.assertFalse(parked)
        self.assertIs(batch_chunked_req, active)
        active.init_next_round_input.assert_called_once_with()
        adder.add_chunked_req.assert_called_once_with(active)

    def test_disabled_feature_keeps_active_chunk_first(self):
        active = _request(prompt_len=4 * self.CHUNK_SIZE)
        scheduler = _scheduler(
            enabled=False, active=active, waiting=[_request(prompt_len=8)]
        )
        adder = MagicMock()
        adder.add_chunked_req.return_value = active

        batch_chunked_req, parked = scheduler._schedule_chunked_req_for_prefill(
            adder, self.CHUNK_SIZE
        )

        self.assertFalse(parked)
        self.assertIs(batch_chunked_req, active)

    def test_oversized_request_does_not_bypass(self):
        active = _request(prompt_len=4 * self.CHUNK_SIZE)
        scheduler = _scheduler(
            enabled=True,
            active=active,
            waiting=[_request(prompt_len=self.CHUNK_SIZE + 1)],
        )
        adder = MagicMock()
        adder.add_chunked_req.return_value = active

        _, parked = scheduler._schedule_chunked_req_for_prefill(adder, self.CHUNK_SIZE)

        self.assertFalse(parked)

    def test_beam_request_does_not_bypass(self):
        active = _request(prompt_len=4 * self.CHUNK_SIZE)
        scheduler = _scheduler(
            enabled=True,
            active=active,
            waiting=[_request(prompt_len=8, beam=True)],
        )
        adder = MagicMock()
        adder.add_chunked_req.return_value = active

        _, parked = scheduler._schedule_chunked_req_for_prefill(adder, self.CHUNK_SIZE)

        self.assertFalse(parked)

    def test_decode_turn_waits_until_decode_is_runnable(self):
        scheduler = _scheduler(enabled=True, active=None, waiting=[])
        scheduler._fair_chunked_prefill_decode_due = True
        running_batch = MagicMock()
        running_batch.is_empty.return_value = True
        running_batch.is_prefill_only = False

        self.assertFalse(
            scheduler._should_run_fair_chunked_prefill_decode(running_batch)
        )
        self.assertTrue(scheduler._fair_chunked_prefill_decode_due)

        running_batch.is_empty.return_value = False
        self.assertTrue(
            scheduler._should_run_fair_chunked_prefill_decode(running_batch)
        )
        self.assertFalse(scheduler._fair_chunked_prefill_decode_due)

    def test_prefill_only_batch_does_not_consume_decode_turn(self):
        scheduler = _scheduler(enabled=True, active=None, waiting=[])
        scheduler._fair_chunked_prefill_decode_due = True
        running_batch = MagicMock()
        running_batch.is_empty.return_value = False
        running_batch.is_prefill_only = True

        self.assertFalse(
            scheduler._should_run_fair_chunked_prefill_decode(running_batch)
        )
        self.assertTrue(scheduler._fair_chunked_prefill_decode_due)


if __name__ == "__main__":
    unittest.main()
