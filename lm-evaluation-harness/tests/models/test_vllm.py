from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from lm_eval import tasks
from lm_eval.api.instance import Instance


task_manager = tasks.TaskManager()


class TestVLLMValidation:
    """Tests for VLLM constructor validation."""

    def test_data_parallel_with_expert_parallel_raises(self):
        """data_parallel_size > 1 with enable_expert_parallel=True must raise."""
        pytest.importorskip("vllm")

        from lm_eval.models.vllm_causallms import VLLM

        with (
            patch.multiple(
                "lm_eval.models.vllm_causallms",
                find_spec=lambda name: None if name == "ray" else MagicMock(),
                LLM=MagicMock(),
                get_tokenizer=MagicMock(return_value=MagicMock()),
            ),
            patch("transformers.AutoConfig.from_pretrained", MagicMock()),
            pytest.raises(ValueError, match=r"data_parallel_size > 1.*expert_parallel"),
        ):
            VLLM(
                pretrained="mock-model",
                data_parallel_size=2,
                enable_expert_parallel=True,
            )

    def test_data_parallel_generate_requires_initialized_ray(self, monkeypatch):
        """Generation dispatch uses the Ray environment initialized by construction."""
        pytest.importorskip("vllm")

        from lm_eval.models import vllm_causallms

        class FakeGenerate:
            def remote(
                self,
                requests: list[list[int]],
                sampling_params: list[object],
                lora_request: object,
            ) -> list[list[int]]:
                return requests

        class FakeActor:
            generate = FakeGenerate()

        init_mock = MagicMock()
        is_initialized_mock = MagicMock(return_value=True)
        fake_ray = SimpleNamespace(
            init=init_mock, is_initialized=is_initialized_mock, get=lambda refs: refs
        )
        monkeypatch.setattr(vllm_causallms, "ray", fake_ray)

        lm = object.__new__(vllm_causallms.VLLM)
        lm.data_parallel_size = 2
        lm.tensor_parallel_size = 1
        lm._data_parallel_replicas = [FakeActor(), FakeActor()]
        lm.lora_request = None

        result = lm._model_generate(
            [[1], [2]],
            generate=True,
            sampling_params=[object(), object()],
        )

        is_initialized_mock.assert_called_once_with()
        init_mock.assert_not_called()
        assert result == [[1], [2]]

    def test_cleanup_kills_data_parallel_actors(self, monkeypatch):
        """Data-parallel Ray actors are explicitly released."""
        pytest.importorskip("vllm")

        from lm_eval.models import vllm_causallms

        actors = [object(), object()]
        kill_mock = MagicMock()
        monkeypatch.setattr(vllm_causallms, "ray", SimpleNamespace(kill=kill_mock))

        lm = object.__new__(vllm_causallms.VLLM)
        lm._data_parallel_replicas = actors

        lm.cleanup()

        kill_mock.assert_has_calls([call(actor) for actor in actors])
        assert lm._data_parallel_replicas is None

    def test_data_parallel_warns_on_insufficient_ray_gpus(self, monkeypatch, caplog):
        """Ray actor creation can queue when GPUs are temporarily unavailable."""
        pytest.importorskip("vllm")

        from lm_eval.models import vllm_causallms

        fake_ray = SimpleNamespace(
            available_resources=MagicMock(return_value={"GPU": 1.0}),
        )
        monkeypatch.setattr(vllm_causallms, "ray", fake_ray)

        lm = object.__new__(vllm_causallms.VLLM)
        lm.data_parallel_size = 2
        lm.tensor_parallel_size = 1

        with caplog.at_level("WARNING"):
            lm._ensure_ray_has_available_gpus()

        assert "Continuing so Ray can queue actors" in caplog.text

    def test_data_parallel_replica_gpu_count_can_be_overridden(self):
        """Ray GPU reservations can differ from vLLM tensor parallel size."""
        pytest.importorskip("vllm")

        from lm_eval.models import vllm_causallms

        lm = object.__new__(vllm_causallms.VLLM)
        lm.tensor_parallel_size = 4

        assert lm._ray_replica_gpu_count() == 4.0

        lm.data_parallel_replica_gpus = 0.5

        assert lm._ray_replica_gpu_count() == 0.5

    def test_parse_logprobs_logs_missing_token(self, caplog):
        """Missing prompt logprobs produce a debug breadcrumb."""
        pytest.importorskip("vllm")

        from lm_eval.models.vllm_causallms import VLLM

        outputs = SimpleNamespace(prompt_logprobs=[None, {30: -1.0}])

        with caplog.at_level("DEBUG"):
            continuation_logprobs, is_greedy = VLLM._parse_logprobs(
                [10, 20],
                outputs,
                ctxlen=1,
            )

        assert continuation_logprobs == -float("inf")
        assert is_greedy is False
        assert "did not include continuation token id 20" in caplog.text

    def test_model_generate_preserves_scalar_sampling_params_for_single_worker(self):
        """Scalar sampling params are passed through to local vLLM batches."""
        pytest.importorskip("vllm")

        from lm_eval.models import vllm_causallms

        captured = {}

        class FakeModel:
            def generate(self, prompts, *, sampling_params, use_tqdm, lora_request):
                captured["sampling_params"] = sampling_params
                return prompts

        lm = object.__new__(vllm_causallms.VLLM)
        lm.data_parallel_size = 1
        lm.model = FakeModel()
        lm.batch_size = "auto"
        lm.lora_request = None
        sampling_params = SimpleNamespace(stop=["stop"])

        lm._model_generate(
            [[1], [2]],
            generate=True,
            sampling_params=sampling_params,
        )

        assert captured["sampling_params"] is sampling_params

    def test_data_parallel_generate_copies_scalar_sampling_params(self, monkeypatch):
        """Scalar sampling params are copied per request before Ray distribution."""
        pytest.importorskip("vllm")

        from lm_eval.models import vllm_causallms

        captured = []

        class FakeGenerate:
            def remote(
                self,
                requests: list[list[int]],
                sampling_params: list[object],
                lora_request: object,
            ) -> list[list[int]]:
                captured.append(sampling_params)
                return requests

        class FakeActor:
            generate = FakeGenerate()

        fake_ray = SimpleNamespace(
            is_initialized=MagicMock(return_value=True), get=lambda refs: refs
        )
        monkeypatch.setattr(vllm_causallms, "ray", fake_ray)

        lm = object.__new__(vllm_causallms.VLLM)
        lm.data_parallel_size = 2
        lm.tensor_parallel_size = 1
        lm._data_parallel_replicas = [FakeActor(), FakeActor()]
        lm.lora_request = None
        sampling_params = SimpleNamespace(stop=["stop"])

        result = lm._model_generate(
            [[1], [2]],
            generate=True,
            sampling_params=sampling_params,
        )

        assert result == [[1], [2]]
        assert len(captured) == 2
        assert captured[0][0] is not sampling_params
        assert captured[1][0] is not sampling_params
        assert captured[0][0] is not captured[1][0]

    def test_thinking_stop_sequences_keep_only_prompt_suffix_stops_for_postprocess(
        self,
    ):
        """Thinking mode only filters stops that end the prompt."""
        pytest.importorskip("vllm")

        from lm_eval.models.vllm_causallms import _thinking_mode_stop_sequences

        stop_sequences = _thinking_mode_stop_sequences(
            prompt="Question\n\nAnswer",
            until=["<eos>", "\n\n", "<DONE>"],
            eos="<eos>",
        )

        assert stop_sequences == ["<eos>", "\n\n", "<DONE>"]

        suffix_stop_sequences = _thinking_mode_stop_sequences(
            prompt="Question\n\n",
            until=["<eos>", "\n\n", "<DONE>"],
            eos="<eos>",
        )

        assert suffix_stop_sequences == ["<eos>", "<DONE>"]


@pytest.mark.skip(reason="requires CUDA")
class Test_VLLM:
    LM = None
    # torch.use_deterministic_algorithms(True)
    task_list = task_manager.load(["arc_easy", "gsm8k", "wikitext"])["tasks"]
    multiple_choice_task = task_list["arc_easy"]  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: list[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances

    # TODO: make proper tests
    def test_logliklihood(self) -> None:
        res = self.LM.loglikelihood(self.MULTIPLE_CH)
        assert len(res) == len(self.MULTIPLE_CH)
        for x in res:
            assert isinstance(x[0], float)

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until)
        assert len(res) == len(self.generate_until)
        for x in res:
            assert isinstance(x, str)

    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        for x in res:
            assert isinstance(x, float)

    def test_loglikelihood_temporarily_disables_enable_thinking(self) -> None:
        with patch.object(self.LM, "enable_thinking", True):
            res = self.LM.loglikelihood(self.MULTIPLE_CH)
            assert self.LM.enable_thinking is True
        assert len(res) == len(self.MULTIPLE_CH)
