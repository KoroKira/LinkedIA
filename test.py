from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Charger le modèle pré-entrainé GPT-3.5
model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

def generer_reponse(entrée_utilisateur, model, tokenizer):
    # Prétraiter l'entrée utilisateur
    entree_encodée = tokenizer.encode(entrée_utilisateur, return_tensors="pt")

    # Générer une réponse en utilisant le modèle
    sortie = model.generate(entree_encodée, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Décode la sortie en texte
    reponse = tokenizer.decode(sortie[0], skip_special_tokens=True)

    return reponse

# Boucle de dialogue
while True:
    entree_utilisateur = input("Vous: ")

    # Générer la réponse
    reponse = generer_reponse(entree_utilisateur, model, tokenizer)

    # Afficher la réponse
    print("IA:", reponse)
