import torch
from torch import nn

from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.dcfe import DCFE
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.shm import SHM
from neuralhydrology.modelzoo.snowdcfe import SnowDCFE
from neuralhydrology.utils.config import Config


class HybridModel(BaseModel):
    """Combine a deep learning model with conceptual hydrological models.

    The deep learning part is selected with the config argument `conceptual_param_network`:
    - 'lstm' (default): an LSTM reads the dynamic inputs and predicts one parameter set per timestep.
    - 'mlp': an MLP reads the static attributes and predicts one parameter set per basin.
    The conceptual model is configurable using the config argument `conceptual_model` (SHM, dCFE, SnowDCFE).

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)

        self.conceptual_model = HybridModel._get_conceptual_model(cfg=cfg)
        n_params = len(self.conceptual_model.parameter_ranges)
        self.param_network = cfg.conceptual_param_network.lower()  # a string, either "mlp" or "lstm"

        if self.param_network == "mlp":
            n_statics = len(cfg.static_attributes)
            if n_statics == 0:
                raise ValueError("conceptual_param_network: mlp requires static_attributes in the config.")
            if cfg.conceptual_param_config != "dynamic":
                # the MLP output is already constant in time, so every averaging scheme is a no-op
                raise ValueError(
                    f"conceptual_param_config '{cfg.conceptual_param_config}' has no effect with "
                    "conceptual_param_network: mlp (parameters are already static). Set it to 'dynamic'."
                )

            self.param_mlp = FC(
                input_size=n_statics,
                hidden_sizes=cfg.conceptual_mlp_hiddens + [n_params],
                activation="tanh",
                dropout=cfg.output_dropout,
            )
        elif self.param_network == "lstm":
            self.embedding_net = InputLayer(cfg)
            self.lstm = nn.LSTM(input_size=self.embedding_net.output_size, hidden_size=cfg.hidden_size)
            self.linear = nn.Linear(in_features=cfg.hidden_size, out_features=n_params)
        else:
            raise NotImplementedError(
                f"conceptual_param_network '{cfg.conceptual_param_network}' not supported. Use 'lstm' or 'mlp'."
            )

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.param_network == "lstm" and self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size : 2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Perform a forward pass on the model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs, parameters and intermediate states coming from the conceptual model
        """
        # TODO: Make the conceptual model take a dict of inputs rather than a concatenated tensor.
        x_conceptual = torch.cat(
            [data["x_d"][k][:, self.cfg.warmup_period :, :] for k in self.cfg.dynamic_conceptual_inputs], axis=-1
        )

        if self.param_network == "mlp":
            raw_params = self.param_mlp(data["x_s"])  # [batch, n_params]
            param_out = raw_params.unsqueeze(1).expand(-1, x_conceptual.shape[1], -1)
        else:
            x_d = self.embedding_net(data)  # [seq, batch, features]
            lstm_output, _ = self.lstm(input=x_d)
            lstm_output = lstm_output.transpose(0, 1)  # [batch, seq, hidden]
            param_out = self.linear(lstm_output[:, self.cfg.warmup_period :, :])

        if self.cfg.conceptual_model.lower() in ["dcfe", "snowdcfe"]:
            pred = self.conceptual_model(
                x_conceptual=x_conceptual,
                lstm_out=param_out,  # TODO: "lstm_out" should eventually be something like "nn_out" or "param_out" but we are just doing a simple test here.
                additional_features=data["static_conceptual_params"],
            )
        else:
            pred = self.conceptual_model(x_conceptual=x_conceptual, lstm_out=param_out)

        return pred

    @staticmethod
    def _get_conceptual_model(cfg: Config) -> BaseConceptualModel:
        """Get conceptual model, depending on the run configuration.

        Parameters
        ----------
        cfg : Config
            The run configuration.

        Returns
        -------
        BaseConceptualModel
            A new conceptual model instance of the type specified in the config.
        """
        if cfg.conceptual_model.lower() == "shm":
            conceptual_model = SHM(cfg=cfg)
        elif cfg.conceptual_model.lower() == "dcfe":
            conceptual_model = DCFE(cfg=cfg)
        elif cfg.conceptual_model.lower() == "snowdcfe":
            conceptual_model = SnowDCFE(cfg=cfg)
        else:
            raise NotImplementedError(f"{cfg.conceptual_model} not implemented or not linked in `_get_conceptual_model()`")

        return conceptual_model
