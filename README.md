# RAG multimodal agentique — assistant de maintenance

CLIP (texte + image, espace commun) → ChromaDB → `retriever_tool` → agent ReAct **LangGraph**.
**CLIP, le LLM (Qwen 2.5 7B) et le VLM (Qwen2.5-VL 7B) tournent sur Google Colab** (GPU) ; le PC exécute le notebook, ChromaDB et l'agent.

```
PC : notebook RAG + ChromaDB + agent LangGraph ──HTTPS (2 tunnels)──► Colab : Ollama (Qwen, Qwen-VL) + serveur CLIP (FastAPI)
```
**Donnée d'entrée : un seul PDF** (démonstration : *Attention Is All You Need*) ; texte et images sont tous extraits de ce PDF. Requêtes du technicien : texte, image, ou image + texte.

## Contenu
| Chemin | Rôle |
|---|---|
| `notebooks/RAG_multimodal_agentique.ipynb` | **Le projet** : tout le code et les étapes 0 à 7 dans un seul notebook |
| `colab/colab_llm_server.ipynb` | À lancer sur Colab : Ollama (Qwen2.5, Qwen2.5-VL) + serveur d'embeddings CLIP + 2 tunnels publics |
| `data/pdfs/` | Le PDF source (`attention_is_all_you_need.pdf`) : seule donnée d'entrée |
| `notebooks/reference_cours.ipynb` | Notebook du cours (référence) |
| `.env.example` | Modèle du fichier `.env` (URL du serveur Colab) |

## Démarrage
1. `pip install -r requirements.txt` (pas besoin de torch sur le PC)
2. Ouvrir `colab/colab_llm_server.ipynb` sur Colab (GPU T4), tout exécuter, copier le bloc `.env` affiché à la fin dans un fichier `.env` à la racine (modèle : `.env.example`). Il contient les 2 URL de tunnel et la clé API du serveur CLIP.
3. Ouvrir `notebooks/RAG_multimodal_agentique.ipynb` et exécuter les cellules dans l'ordre.

Sans Colab : laisser `EMBED_BASE_URL` vide (CLIP local, nécessite `torch` + `open-clip-torch`) et mettre `USE_VLM = False` à l'étape 3.

## Ce qui est stocké dans ChromaDB
| | Texte | Image |
|---|---|---|
| `document` | le fragment de texte | la **description** de l'image, générée par Qwen2.5-VL |
| `embedding` | vecteur CLIP du texte | vecteur CLIP de l'**image** |
| `metadata` | type, pdf, page, chunk, document_id, source | type, pdf, page, origin, **`image_path`** (chemin sur le PC), document_id |

## Choix de conception
- **CLIP sur Colab** pour l'indexation et la recherche (texte et images dans le même espace) : le serveur renvoie des vecteurs déjà normalisés et découpe le texte en fragments ; l'API est protégée par une clé (`x-api-key`) ; **Qwen** comme cerveau de l'agent ; **Qwen2.5-VL** pour décrire les images à l'indexation.
- Image + texte : les deux vecteurs CLIP sont fusionnés (moyenne renormalisée).
- L'outil ne génère rien : il renvoie des observations brutes (passages, descriptions, chemins d'images, scores) ; le LLM rédige la réponse.
- Requêtes de recherche rédigées **en anglais** par l'agent (CLIP est surtout anglophone), réponse en français.
- Seules 3 images sont intégrées au PDF ; les pages sont donc aussi rendues en PNG pour couvrir les figures vectorielles.

