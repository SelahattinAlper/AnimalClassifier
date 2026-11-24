
from datasets import load_dataset, DatasetDict

# Veri setini indir
dataset = load_dataset(
    "Rapidata/Animals-10",
    split="train"
)

print("Toplam görüntü:", len(dataset))
print("Sınıflar:", dataset.features["label"].names)

# %80 eğitim, %20 geçici ayrım
first_split = dataset.train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

# Kalan %20'yi doğrulama ve teste ayır
second_split = first_split["test"].train_test_split(
    test_size=0.5,
    seed=42,
    stratify_by_column="label"
)

splits = DatasetDict({
    "train": first_split["train"],
    "validation": second_split["train"],
    "test": second_split["test"]
})

# Proje klasörüne kaydet
splits.save_to_disk("data/animals10")

for name, subset in splits.items():
    print(f"{name}: {len(subset)} görüntü")

print("Veri seti hazır!")
