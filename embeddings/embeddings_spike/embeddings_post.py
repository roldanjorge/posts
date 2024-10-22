from embeddings_utils import cosine_similarity, get_embedding
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())
client = OpenAI()

def compare_strings(string1, string2):
    embed1 = get_embedding(string1, model="text-embedding-3-small")
    embed2 = get_embedding(string2, model="text-embedding-3-small")
    similarity = cosine_similarity(embed1, embed2)
    print(f"{string1}:{string2} = {similarity}")


if __name__ == '__main__':
    compare_strings(string1="hi", string2="bye")
    compare_strings(string1="ugly", string2="beautiful")
    compare_strings(string1="I am good", string2="I am bad")
    compare_strings(string1="My favorite book is LOL1", string2="My favorite book is allalalal")

