# test.py
# 動作確認用スクリプト
# 自身のCUDAに合ったtorchを入れる必要あり

from comet.models import download_model
model = download_model("wmt-large-da-estimator-1719")
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
results = model.predict(data, cuda=True, show_progress=True)
print(results)