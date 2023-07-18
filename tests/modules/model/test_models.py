# from nhssynth.modules.model import MODELS


# def test_models(transformed) -> None:
#     for _, MODEL in MODELS.items():
#         model = MODEL(transformed, single_column_indices=[i for i, _ in enumerate(transformed.columns)])
#         model.train(num_epochs=5, patience=1)
#         synth = model.generate()
#         assert synth.shape == transformed.shape
#         assert synth.columns.tolist() == transformed.columns.tolist()
