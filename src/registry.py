"""
Identity registry: Stores and retrieves face embeddings keyed by name.
Persists to disk as a .npz file and loads on import
"""

import numpy as np
import os
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

REGISTRY_PATH = config["paths"]["registry"]
THRESHOLD = config["recognition"]["threshold"]

names = []
embeddings = []

def load():
    """Load registry from disk to memory"""
    if os.path.exists(REGISTRY_PATH):
        data = np.load(REGISTRY_PATH)
        names.clear()
        embeddings.clear()
        names.extend(list(data["names"]))
        embeddings.extend(list(data["embeddings"]))

def save():
    """Save current registry to disk"""
    if names:
        np.savez(REGISTRY_PATH, names=np.array(names), embeddings=np.array(embeddings))
    elif os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)

def register(name, embedding):
    """Add or modify identity"""
    if name in names:
        index = names.index(name)
        embeddings[index] = embedding
    else:
        names.append(name)
        embeddings.append(embedding)
    save()

def search(embedding):
    """
    Return closest matching embeddings pair as:
    {"name", "distance"} or None
    """
    if not names:
        return None
    
    distances = [np.linalg.norm(embedding - e) for e in embeddings]
    best_index = int(np.argmin(distances))

    if distances[best_index] <= THRESHOLD:
        return {"name": names[best_index], "distance": distances[best_index]}
    
    return None

def delete(name):
    """Remove an identity. Return false if not exist."""
    if name not in names:
        return False
    
    index = names.index(name)
    names.pop(index)
    embeddings.pop(index)
    save()
    return True


def list_names():
    """list names..."""
    return list(names)


load()