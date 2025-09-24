def test_imports():
    import src.data as data
    import src.models as models
    import src.train as train
    assert hasattr(models, "svm_rbf")
