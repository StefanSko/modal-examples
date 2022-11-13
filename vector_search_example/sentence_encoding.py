import modal

stub = modal.Stub("sentence-encoding")

@stub.function(
    image=modal.Image.debian_slim().pip_install(["sentence-transformers==2.0.0"]),
    retries=3,
)
def encode_sentence(sentence):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    return model.encode(sentence)
