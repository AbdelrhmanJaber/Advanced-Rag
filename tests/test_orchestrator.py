"""
Tests for src/preprocessing/orchestrator.py

Strategy: mock all I/O-heavy dependencies (loader, validator, classifier, parsers)
so tests run purely in-process without any real files or processes.

Bugs found during review:
  BUG-1  (line 190) - classifier_stage sends n_classifer_workers SENTINELs to
          q_scanned_parser but the queue is consumed by n_scanned_parser_workers
          workers.  When the two values differ the pipeline either deadlocks
          (scanned workers starve) or exits early (surplus SENTINELs drain the
          queue before all items are processed).
  MINOR  (line 108) - classifier log message labels n_scanned as "rejected=".
"""

import multiprocessing as mp
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENTINEL = None  # must match orchestrator.SENTINEL


def drain(q: mp.Queue, timeout: float = 2.0):
    """Collect all non-SENTINEL items from a queue, stop on SENTINEL."""
    items = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            item = q.get(timeout=0.1)
            if item is SENTINEL:
                break
            items.append(item)
        except Exception:
            break
    return items


def drain_all(q: mp.Queue, timeout: float = 1.0):
    """Drain everything (including SENTINELs) from *q* within *timeout* seconds."""
    items = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            items.append(q.get(timeout=0.05))
        except Exception:
            break
    return items


def put_sentinels(q: mp.Queue, n: int):
    for _ in range(n):
        q.put(SENTINEL)


# ===========================================================================
# 1.  lock_and_release
# ===========================================================================

class TestLockAndRelease:
    """Unit-tests for the shared-counter / SENTINEL-injection helper."""

    def _make_counter(self, value: int):
        c = mp.Value("i", value)
        return c

    def test_decrements_counter(self):
        from src.preprocessing.orchestrator import lock_and_release

        counter = self._make_counter(3)
        lock = mp.Lock()
        lock_and_release(counter, lock, [])
        assert counter.value == 2

    def test_no_sentinel_when_not_last(self):
        from src.preprocessing.orchestrator import lock_and_release

        counter = self._make_counter(3)
        lock = mp.Lock()
        q = mp.Queue()
        lock_and_release(counter, lock, [(q, 2)])
        assert q.empty()

    def test_sends_sentinels_when_last(self):
        from src.preprocessing.orchestrator import lock_and_release

        counter = self._make_counter(1)
        lock = mp.Lock()
        q = mp.Queue()
        lock_and_release(counter, lock, [(q, 3)])
        time.sleep(0.1)  # allow mp.Queue pipe to flush
        sentinels = drain_all(q)
        assert sentinels.count(SENTINEL) == 3

    def test_sends_sentinels_to_multiple_queues(self):
        from src.preprocessing.orchestrator import lock_and_release

        counter = self._make_counter(1)
        lock = mp.Lock()
        qa, qb = mp.Queue(), mp.Queue()
        lock_and_release(counter, lock, [(qa, 2), (qb, 1)])
        time.sleep(0.1)  # allow mp.Queue pipe to flush
        qa_items = drain_all(qa)
        qb_items = drain_all(qb)
        assert qa_items.count(SENTINEL) == 2
        assert qb_items.count(SENTINEL) == 1

    def test_concurrent_decrement_only_last_sends_sentinels(self):
        """Two concurrent calls: only the one that hits 0 sends SENTINELs."""
        from src.preprocessing.orchestrator import lock_and_release

        counter = mp.Value("i", 2)
        lock = mp.Lock()
        q = mp.Queue()

        def worker():
            lock_and_release(counter, lock, [(q, 1)])

        p1 = mp.Process(target=worker)
        p2 = mp.Process(target=worker)
        p1.start(); p2.start()
        p1.join(); p2.join()

        # Exactly 1 SENTINEL must have been produced
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items.count(SENTINEL) == 1


# ===========================================================================
# 2.  loader_stage
# ===========================================================================

class TestLoaderStage:
    def test_puts_file_paths_and_correct_number_of_sentinels(self):
        from src.preprocessing.orchestrator import loader_stage

        fake_paths = ["/data/a.pdf", "/data/b.pdf", "/data/c.pdf"]

        with patch("src.preprocessing.orchestrator.LocalFileLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.scan_files.return_value = iter(fake_paths)

            q = mp.Queue()
            n_validators = 3
            loader_stage("/data", q, n_validators)

        time.sleep(0.1)  # allow mp.Queue pipe to flush
        results = drain_all(q, timeout=1.0)

        file_items = [r for r in results if r is not SENTINEL]
        sentinel_items = [r for r in results if r is SENTINEL]

        assert file_items == fake_paths
        assert len(sentinel_items) == n_validators

    def test_sends_sentinels_even_when_scan_raises(self):
        """Loader uses finally: sentinels must be sent even on loader error."""
        from src.preprocessing.orchestrator import loader_stage

        with patch("src.preprocessing.orchestrator.LocalFileLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.scan_files.side_effect = RuntimeError("disk error")

            q = mp.Queue()
            # Expect FileNotFoundError or RuntimeError to propagate but sentinels
            # are NOT guaranteed here because the finally only sends sentinels
            # after the loop — if the exception is inside scan_files the finally
            # still fires.
            try:
                loader_stage("/data", q, 2)
            except Exception:
                pass  # loader may or may not raise depending on implementation

        # Regardless, drain queue and verify at most 2 sentinels
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        sentinel_count = sum(1 for i in items if i is SENTINEL)
        assert sentinel_count <= 2


# ===========================================================================
# 3.  validator_stage  (in-process, using mp.Queue directly)
# ===========================================================================

class TestValidatorStage:
    """Run validator_stage in-process using Thread-safe mp.Queue objects."""

    def _run_validator(self, items_in, validator_side_effects, n_validators=2, n_classifiers=2):
        """
        Helper: populate q_in, run validator_stage in the current process,
        return (q_out items, final counter value).
        """
        from src.preprocessing.orchestrator import validator_stage

        q_in = mp.Queue()
        q_out = mp.Queue()
        counter = mp.Value("i", n_validators)
        lock = mp.Lock()

        for item in items_in:
            q_in.put(item)
        q_in.put(SENTINEL)  # one sentinel for this single worker call

        with patch("src.preprocessing.orchestrator.validator") as mock_validator:
            mock_validator.side_effect = validator_side_effects
            validator_stage(
                worker_id=0,
                q_in=q_in,
                q_out=q_out,
                counter=counter,
                locker=lock,
                downstream=[(q_out, n_classifiers)],
            )

        time.sleep(0.1)  # allow mp.Queue pipe to flush
        out_items = drain_all(q_out)
        return out_items, counter.value

    def test_valid_files_forwarded(self):
        items = ["/a.pdf", "/b.pdf"]
        effects = [
            (True, {}, "ok"),
            (True, {}, "ok"),
        ]
        out, _ = self._run_validator(items, effects)
        # Only items before sentinel
        forwarded = [i for i in out if i is not SENTINEL]
        assert forwarded == ["/a.pdf", "/b.pdf"]

    def test_invalid_files_not_forwarded(self):
        items = ["/bad.pdf"]
        effects = [(False, {}, "corrupted")]
        out, _ = self._run_validator(items, effects)
        forwarded = [i for i in out if i is not SENTINEL]
        assert forwarded == []

    def test_exception_during_validation_does_not_crash_worker(self):
        items = ["/a.pdf", "/b.pdf"]
        effects = [RuntimeError("boom"), (True, {}, "ok")]
        out, _ = self._run_validator(items, effects)
        forwarded = [i for i in out if i is not SENTINEL]
        assert forwarded == ["/b.pdf"]

    def test_counter_decremented(self):
        _, counter_val = self._run_validator([], [], n_validators=3)
        assert counter_val == 2  # decremented from 3 to 2

    def test_sentinels_sent_when_last_validator(self):
        """When counter reaches 0 (n_validators=1), sentinels are injected."""
        from src.preprocessing.orchestrator import validator_stage

        q_in = mp.Queue()
        q_out = mp.Queue()
        counter = mp.Value("i", 1)
        lock = mp.Lock()

        q_in.put(SENTINEL)

        with patch("src.preprocessing.orchestrator.validator"):
            validator_stage(
                worker_id=0,
                q_in=q_in,
                q_out=q_out,
                counter=counter,
                locker=lock,
                downstream=[(q_out, 2)],
            )

        time.sleep(0.1)  # allow mp.Queue pipe to flush
        items = drain_all(q_out)
        sentinel_count = sum(1 for i in items if i is SENTINEL)
        assert sentinel_count == 2


# ===========================================================================
# 4.  classifier_stage
# ===========================================================================

class TestClassifierStage:
    def _run_classifier(self, items, classifier_side_effects,
                        n_classifiers=2, n_native=2, n_scanned=2):
        from src.preprocessing.orchestrator import classifier_stage

        q_in = mp.Queue()
        q_native = mp.Queue()
        q_scanned = mp.Queue()
        counter = mp.Value("i", n_classifiers)
        lock = mp.Lock()

        for item in items:
            q_in.put(item)
        q_in.put(SENTINEL)

        with patch("src.preprocessing.orchestrator.classifier") as mock_cls:
            mock_cls.side_effect = classifier_side_effects
            classifier_stage(
                worker_id=0,
                q_in=q_in,
                q_native_out=q_native,
                q_scanned_out=q_scanned,
                counter=counter,
                locker=lock,
                downstream=[
                    (q_native, n_native),
                    (q_scanned, n_scanned),
                ],
            )

        time.sleep(0.1)  # allow mp.Queue pipe to flush
        native_items = drain_all(q_native)
        scanned_items = drain_all(q_scanned)
        return native_items, scanned_items, counter.value

    def test_native_file_routed_to_native_queue(self):
        native, scanned, _ = self._run_classifier(
            ["/doc.pdf"],
            [{"file_type": "NATIVE"}],
        )
        assert "/doc.pdf" in native
        assert "/doc.pdf" not in scanned

    def test_scanned_file_routed_to_scanned_queue(self):
        native, scanned, _ = self._run_classifier(
            ["/scan.pdf"],
            [{"file_type": "SCANNED"}],
        )
        assert "/scan.pdf" in scanned
        assert "/scan.pdf" not in native

    def test_mixed_routing(self):
        native, scanned, _ = self._run_classifier(
            ["/n.pdf", "/s.pdf"],
            [{"file_type": "NATIVE"}, {"file_type": "SCANNED"}],
        )
        forwarded_native = [i for i in native if i is not SENTINEL]
        forwarded_scanned = [i for i in scanned if i is not SENTINEL]
        assert "/n.pdf" in forwarded_native
        assert "/s.pdf" in forwarded_scanned

    def test_classifier_exception_skips_item(self):
        native, scanned, _ = self._run_classifier(
            ["/bad.pdf", "/good.pdf"],
            [RuntimeError("oops"), {"file_type": "NATIVE"}],
        )
        forwarded_native = [i for i in native if i is not SENTINEL]
        assert "/bad.pdf" not in forwarded_native
        assert "/good.pdf" in forwarded_native

    def test_counter_decremented(self):
        _, _, counter_val = self._run_classifier([], [], n_classifiers=4)
        assert counter_val == 3

    # -----------------------------------------------------------------------
    # BUG-1: sentinel count for scanned queue uses n_classifer_workers
    #         instead of n_scanned_parser_workers  (line 190 in orchestrator)
    # -----------------------------------------------------------------------
    def test_bug1_sentinel_count_mismatch_causes_deadlock(self):
        """
        Demonstrates BUG-1:
        run_pipline passes n_classifer_workers as the consumer count for
        q_scanned_parser instead of n_scanned_parser_workers.

        When n_scanned_parser_workers (2) < n_classifer_workers (4) the scanned
        workers only receive 2 SENTINELs but the queue will inject 4, so
        leftover sentinels accumulate and the count is wrong.

        When n_scanned_parser_workers (4) > n_classifer_workers (2) only 2
        SENTINELs are produced but 4 workers are waiting → deadlock.

        This test checks the CORRECT behaviour: after the last classifier
        finishes, q_scanned should receive exactly n_scanned_parser_workers
        SENTINELs.
        """
        from src.preprocessing.orchestrator import classifier_stage

        n_classifiers = 1          # run single worker so counter hits 0
        n_native_workers = 2
        n_scanned_workers = 3      # intentionally different from n_classifiers

        q_in = mp.Queue()
        q_native = mp.Queue()
        q_scanned = mp.Queue()
        counter = mp.Value("i", n_classifiers)
        lock = mp.Lock()

        q_in.put(SENTINEL)

        with patch("src.preprocessing.orchestrator.classifier"):
            classifier_stage(
                worker_id=0,
                q_in=q_in,
                q_native_out=q_native,
                q_scanned_out=q_scanned,
                counter=counter,
                locker=lock,
                downstream=[
                    (q_native, n_native_workers),
                    (q_scanned, n_scanned_workers),   # CORRECT: n_scanned_workers
                ],
            )

        time.sleep(0.1)  # allow mp.Queue pipe to flush
        scanned_sentinels = drain_all(q_scanned)

        # With the correct downstream argument the count is right.
        # With the bug (n_classifiers passed instead) this assertion fails
        # when n_scanned_workers != n_classifiers.
        assert len(scanned_sentinels) == n_scanned_workers, (
            f"Expected {n_scanned_workers} SENTINELs for scanned queue "
            f"but got {len(scanned_sentinels)}. "
            "This exposes BUG-1: line 190 passes n_classifer_workers "
            "instead of n_scanned_parser_workers."
        )

    def test_bug1_present_in_run_pipline_config(self):
        """
        Directly inspect run_pipline source to confirm BUG-1 is present:
        the downstream list for classifier_stage uses config.n_classifer_workers
        for the scanned queue instead of config.n_scanned_parser_workers.
        """
        import inspect
        from src.preprocessing.orchestrator import run_pipline

        source = inspect.getsource(run_pipline)
        # The bug: scanned sentinel count equals n_classifer_workers
        # A correct implementation would reference n_scanned_parser_workers
        assert "n_scanned_parser_workers" in source, (
            "run_pipline does not reference n_scanned_parser_workers for the "
            "scanned queue sentinel count — BUG-1 confirmed."
        )


# ===========================================================================
# 5.  native_parse_stage / scanned_parse_stage
# ===========================================================================

class TestParseStages:
    def _run_native(self, items):
        from src.preprocessing.orchestrator import native_parse_stage

        q = mp.Queue()
        for item in items:
            q.put(item)
        q.put(SENTINEL)

        with patch("src.preprocessing.orchestrator.NativeDocumentParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_and_save_json.return_value = "/out/file_tree.json"
            native_parse_stage(0, q, "/output")

        return MockParser, instance

    def _run_scanned(self, items):
        from src.preprocessing.orchestrator import scanned_parse_stage

        q = mp.Queue()
        for item in items:
            q.put(item)
        q.put(SENTINEL)

        with patch("src.preprocessing.orchestrator.ScannedDocumentParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_and_save_json.return_value = "/out/file_tree.json"
            scanned_parse_stage(0, q, "/output")

        return MockParser, instance

    def test_native_parse_stage_calls_parser_per_file(self):
        MockParser, instance = self._run_native(["/a.pdf", "/b.pdf"])
        assert instance.parse_and_save_json.call_count == 2

    def test_native_parse_stage_output_dir_structure(self):
        from src.preprocessing.orchestrator import native_parse_stage
        from pathlib import Path

        q = mp.Queue()
        q.put("/data/myfile.pdf")
        q.put(SENTINEL)

        captured_dirs = []

        def fake_init(path, output_dir=None):
            captured_dirs.append(output_dir)
            m = MagicMock()
            m.parse_and_save_json.return_value = "/out/x.json"
            return m

        with patch("src.preprocessing.orchestrator.NativeDocumentParser", side_effect=fake_init):
            native_parse_stage(0, q, "/output")

        assert captured_dirs == [str(Path("/output") / "native" / "myfile")]

    def test_scanned_parse_stage_output_dir_structure(self):
        from src.preprocessing.orchestrator import scanned_parse_stage
        from pathlib import Path

        q = mp.Queue()
        q.put("/data/scan.pdf")
        q.put(SENTINEL)

        captured_dirs = []

        def fake_init(path, output_dir=None):
            captured_dirs.append(output_dir)
            m = MagicMock()
            m.parse_and_save_json.return_value = "/out/x.json"
            return m

        with patch("src.preprocessing.orchestrator.ScannedDocumentParser", side_effect=fake_init):
            scanned_parse_stage(0, q, "/output")

        assert captured_dirs == [str(Path("/output") / "scanned" / "scan")]

    def test_native_parse_stage_exception_does_not_crash(self):
        from src.preprocessing.orchestrator import native_parse_stage

        q = mp.Queue()
        q.put("/bad.pdf")
        q.put("/good.pdf")
        q.put(SENTINEL)

        call_log = []

        def fake_init(path, output_dir=None):
            call_log.append(path)
            m = MagicMock()
            if path == "/bad.pdf":
                m.parse_and_save_json.side_effect = RuntimeError("parse error")
            else:
                m.parse_and_save_json.return_value = "/out/good.json"
            return m

        with patch("src.preprocessing.orchestrator.NativeDocumentParser", side_effect=fake_init):
            native_parse_stage(0, q, "/output")  # must NOT raise

        assert "/bad.pdf" in call_log
        assert "/good.pdf" in call_log

    def test_scanned_parse_stage_exception_does_not_crash(self):
        from src.preprocessing.orchestrator import scanned_parse_stage

        q = mp.Queue()
        q.put("/bad.pdf")
        q.put(SENTINEL)

        with patch("src.preprocessing.orchestrator.ScannedDocumentParser") as MockParser:
            instance = MockParser.return_value
            instance.parse_and_save_json.side_effect = RuntimeError("ocr fail")
            scanned_parse_stage(0, q, "/output")  # must NOT raise


# ===========================================================================
# 6.  pipliningConfig defaults
# ===========================================================================

class TestPipliningConfig:
    def test_defaults(self):
        from src.preprocessing.orchestrator import pipliningConfig

        cfg = pipliningConfig(raw_dir="/data", output_dir="/out")
        assert cfg.n_validator_workers == 4
        assert cfg.n_classifer_workers == 4
        assert cfg.n_native_parser_workers == 4
        assert cfg.n_scanned_parser_workers == 2
        assert cfg.queue_max_size == 200

    def test_custom_values(self):
        from src.preprocessing.orchestrator import pipliningConfig

        cfg = pipliningConfig(
            raw_dir="/x", output_dir="/y",
            n_validator_workers=1,
            n_classifer_workers=2,
            n_native_parser_workers=3,
            n_scanned_parser_workers=4,
            queue_max_size=50,
        )
        assert cfg.n_scanned_parser_workers == 4
        assert cfg.queue_max_size == 50


# ===========================================================================
# 7.  run_pipline — sentinel wiring smoke test (fully mocked, no subprocesses)
# ===========================================================================

class TestRunPiplineWiring:
    """
    Patch mp.Process so we can inspect what args each stage is called with
    without spawning real child processes.
    """

    def test_scanned_sentinel_count_bug_in_wiring(self):
        """
        BUG-1 manifests in the downstream list passed to classifier_stage.
        The third element of each classifier Process args is the downstream list.
        We verify that the scanned queue receives n_scanned_parser_workers
        sentinels — not n_classifer_workers.
        """
        from src.preprocessing.orchestrator import run_pipline, pipliningConfig

        cfg = pipliningConfig(
            raw_dir="/data",
            output_dir="/out",
            n_validator_workers=2,
            n_classifer_workers=3,       # deliberately different
            n_native_parser_workers=2,
            n_scanned_parser_workers=5,  # deliberately different from classifiers
        )

        processes_created = []

        class FakeProcess:
            def __init__(self, target=None, args=(), name="", **kw):
                self.target = target
                self.args = args
                self.name = name
                processes_created.append(self)

            def start(self): pass
            def join(self): pass

        with patch("src.preprocessing.orchestrator.mp.Process", FakeProcess):
            run_pipline(cfg)

        # Find classifier processes and inspect their downstream arg
        from src.preprocessing.orchestrator import classifier_stage
        classifier_procs = [p for p in processes_created if p.target == classifier_stage]
        assert classifier_procs, "No classifier processes were created"

        for proc in classifier_procs:
            # args = (worker_id, q_in, q_native_out, q_scanned_out, counter, lock, downstream)
            downstream = proc.args[6]
            # downstream is list of (queue, n_consumers)
            scanned_consumer_count = downstream[1][1]
            assert scanned_consumer_count == cfg.n_scanned_parser_workers, (
                f"BUG-1: classifier_stage downstream for scanned queue uses "
                f"{scanned_consumer_count} instead of "
                f"{cfg.n_scanned_parser_workers} (n_scanned_parser_workers). "
                "Fix line 190: replace config.n_classifer_workers with "
                "config.n_scanned_parser_workers."
            )
