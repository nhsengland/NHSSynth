from nhssynth.modules.model import MODELS


def test_models(prepared) -> None:
    for _, MODEL in MODELS.items():
        model = MODEL(prepared, singles=[i for i, _ in enumerate(prepared.columns)])
        model.train(num_epochs=5, patience=1)
        synth = model.generate()
        assert synth.shape == prepared.shape
        assert synth.columns.tolist() == prepared.columns.tolist()
