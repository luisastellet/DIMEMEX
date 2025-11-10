import json

# Esse aqui rodei no notebook sem colab, então os paths não fazem referência ao drive

def juntar_jsons(pasta):

    with open(f'{pasta}/{pasta}_data.json', 'r', encoding='utf-8') as file:
        data_original = json.load(file)
    with open(f'{pasta}/{pasta}_data_translated.json', 'r', encoding='utf-8') as file:
        data_traduzido = json.load(file)

    new_data = []
    for indice in range(len(data_original)):
        result = {
            "MEME-ID": data_original[indice]["MEME-ID"],
            "original": {
                "text": data_original[indice]["text"],
                "description": data_original[indice]["description"],
            },
            "traduzido" : {
                "text": data_traduzido[indice]["text"],
                "description": data_traduzido[indice]["description"],
            }
        }

        new_data.append(result)
        print(f"Item {indice+1} \n data traduzido: {result['traduzido']} \n")

    with open(f'{pasta}/{pasta}_data_junto.json', 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    print(f'Arquivos juntados e salvos em {pasta}/{pasta}_data_junto.json')

if __name__ == "__main__":
    for pasta in ["test", "validation", "train"]:
        juntar_jsons(pasta)