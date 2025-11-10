# DIMEMEX
Trabalho realizado durante a disciplina Modelos de linguagem neurais no semestre 2025/2 que foi lecionada pela professora Aline Paes.

- A ideia deste trabalho é fazer a tradução dos memes que estão em espanhol para o português para analisarmos se memes que são ofensivos e/ou possuem discurso de ódio mantêm essas características quando traduzidos para o português.

**Tarefa 1:**
Traduzir as frases para o português e analisar se a tradução foi coerente junto de uma análise humana.

**Tarefa 2:**
Análise de discurso de ódio quanto a existência ou não dele na frase (hate speech, inappropriate content, and neither.).

Combinações para o fine-tuning (Faremos 1 na língua portuguesa e 1 no espanhol, logo 2 ajustes para cada combinação, sendo um em cada língua):

1) Texto (Bárbara)

2) Texto e descrição (Amanda)

3) Texto, descrição e imagem (Juan)

4) Imagem (Luisa)


**Sobre a 1ª tarefa:**
- O modelo usado para a tradução do text e description será google/gemma-3-4b-it
- As métricas que aplicaremos para analisar a tradução serão: bleurt, bertscore, cometkiwi e chrf (Todas sem referência)

**Sobre a 2ª tarefa:**
- O modelo que usaremso no fine-tuning será HuggingFaceTB/SmolVLM-256M-Instruct
- Pós o ajuste fino faremos inferência com o modelo ajustado e ele sem o ajuste para analisarmos as métricas
