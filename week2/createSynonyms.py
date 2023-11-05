import fasttext

TOP_WORDS_PATH = "/workspace/datasets/fasttext/top_words.txt" 
SYNONYMS_PATH = "/workspace/datasets/fasttext/synonyms.csv"
MODEL_PATH = "/workspace/datasets/fasttext/title_model_100_epoch25.bin"
THRESHOLD = 0.75

model = model = fasttext.load_model(MODEL_PATH)

with open(TOP_WORDS_PATH) as f_in, open(SYNONYMS_PATH, "w") as f_out:
  for word in f_in:
    line = ""
    word = word.strip()
    if len(line) == 0:
      line = word

    for similarity, synomym in model.get_nearest_neighbors(word):
        if similarity > THRESHOLD:
          line += f",{synomym}"
    f_out.write(line + "\n")