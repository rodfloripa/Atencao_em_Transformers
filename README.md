
<p align="justify"><h1>Visualizando Self-Attention em Transformers</h1></p>

<p align="justify">

Este projeto implementa uma versão educacional do mecanismo de <b>Self-Attention</b>, núcleo matemático utilizado em arquiteturas modernas como GPT, BERT, LLaMA e Gemini.

O objetivo não é apenas utilizar um Transformer pronto, mas sim mostrar detalhadamente:

- como os embeddings são criados;
- como Queries, Keys e Values funcionam;
- como a matriz de atenção é calculada;
- como os pesos de atenção são distribuídos;
- como o modelo aprende relações semânticas entre palavras.

Além disso, o projeto gera um <b>heatmap interpretável</b>, permitindo visualizar para quais tokens cada palavra está “olhando”.

Esse tipo de projeto possui excelente valor para portfólio porque demonstra domínio de:

- Deep Learning;
- NLP;
- Transformers;
- Álgebra Linear;
- PyTorch;
- Interpretabilidade de modelos.

</p>

---



<p align="justify"><h1>Funcionamento Geral</h1></p>

<p align="justify">

O pipeline do projeto segue exatamente o fluxo interno utilizado em Transformers reais.

A execução ocorre nas seguintes etapas:

1. Tokenização da frase;
2. Conversão para IDs numéricos;
3. Criação dos embeddings;
4. Projeção em:

   * Query (Q)
   * Key (K)
   * Value (V)
5. Cálculo dos scores de atenção;
6. Aplicação do Softmax;
7. Geração da saída ponderada;
8. Visualização do mapa de atenção.

</p>

---

<p align="justify"><h1>Fórmula Principal da Self-Attention</h1></p>

```math
Attention(Q,K,V)=softmax((QKᵀ)/√d_k)V
```

<p align="justify">

Essa fórmula representa o mecanismo central dos Transformers.

Ela mede a similaridade entre Queries e Keys utilizando produto escalar.

O resultado é normalizado pelo Softmax para gerar probabilidades de atenção.

Depois, essas probabilidades ponderam os Values, produzindo a saída final contextualizada.

</p>

---

<p align="justify"><h1>1. Tokenização</h1></p>

```python
sentence = "a inteligencia artificial aprende padroes"

tokens = sentence.split()
```

<p align="justify">

Aqui a frase é dividida em palavras individuais.

Cada palavra será tratada como um token independente pelo Transformer.

Exemplo:

</p>

```python
['a', 'inteligencia', 'artificial', 'aprende', 'padroes']
```

<p align="justify">

Em modelos reais, a tokenização costuma ser mais sofisticada, utilizando técnicas como:

* Byte Pair Encoding (BPE);
* SentencePiece;
* WordPiece.

Neste projeto, o split simples facilita a interpretação visual do funcionamento interno do modelo.

</p>

---

<p align="justify"><h1>2. Criação do Vocabulário</h1></p>

```python
vocab = {word: idx for idx, word in enumerate(tokens)}

token_ids = torch.tensor([vocab[t] for t in tokens])
```

<p align="justify">

Transformers não trabalham diretamente com texto.

Cada palavra precisa ser convertida em um índice numérico.

Exemplo:

</p>

```python
{
 'a': 0,
 'inteligencia': 1,
 'artificial': 2,
 'aprende': 3,
 'padroes': 4
}
```

<p align="justify">

Esses IDs serão usados pela camada de embedding.

</p>

---

<p align="justify"><h1>3. Embeddings</h1></p>

```python
embedding = nn.Embedding(len(vocab), embedding_dim)

x = embedding(token_ids)
```

<p align="justify">

Embeddings transformam tokens em vetores densos de números reais.

Esses vetores representam características semânticas das palavras.

Exemplo:

</p>

```python
'inteligencia' → [0.12, -0.45, 0.88, ...]
```

<p align="justify">

Modelos reais utilizam embeddings com centenas ou milhares de dimensões.

Neste projeto utilizamos dimensão reduzida para facilitar visualização e entendimento.

</p>

---

<p align="justify"><h1>4. Classe SelfAttention</h1></p>

```python
class SelfAttention(nn.Module):
```

<p align="justify">

Essa classe implementa o núcleo matemático do Transformer.

Ela cria três projeções lineares:

* Query;
* Key;
* Value.

</p>

```python
self.query = nn.Linear(embed_dim, embed_dim)
self.key = nn.Linear(embed_dim, embed_dim)
self.value = nn.Linear(embed_dim, embed_dim)
```

<p align="justify">

Essas camadas aprendem representações diferentes do mesmo token.

</p>

---

<p align="justify"><h1>5. Geração de Q, K e V</h1></p>

```python
Q = self.query(x)
K = self.key(x)
V = self.value(x)
```

<p align="justify">

Cada embedding é projetado em três espaços diferentes:

* Query → o que o token procura;
* Key → o que o token oferece;
* Value → informação carregada.

O Transformer compara Queries com Keys para medir relevância contextual.

</p>

---

<p align="justify"><h1>6. Attention Scores</h1></p>

```python
scores = torch.matmul(Q, K.T) / np.sqrt(dk)
```

<p align="justify">

Aqui ocorre o cálculo principal da atenção.

O produto escalar mede similaridade entre tokens.

Quanto maior o score:

* maior a relação semântica;
* maior será a atenção atribuída.

A divisão por raiz quadrada de d_k evita crescimento excessivo dos valores, estabilizando o treinamento.

</p>

```math
scores = (QKᵀ)/√d_k
```

---

<p align="justify"><h1>7. Softmax</h1></p>

```python
attention_weights = torch.softmax(scores, dim=-1)
```

<p align="justify">

O Softmax transforma os scores em probabilidades.

Cada linha da matriz passa a somar 1.

Isso permite interpretar os valores como níveis de atenção entre palavras.

</p>

```math
AttentionWeights = softmax(scores)
```

---

<p align="justify"><h1>8. Saída Final</h1></p>

```python
output = torch.matmul(attention_weights, V)
```

<p align="justify">

Os pesos de atenção são usados para combinar os Values.

Isso gera uma nova representação contextualizada dos tokens.

Em outras palavras:

cada palavra passa a carregar informações relevantes das demais palavras da frase.

</p>

```math
Output = AttentionWeights × V
```

---

<p align="justify"><h1>9. Heatmap de Atenção</h1></p>

```python
sns.heatmap(
    weights_np,
    annot=True,
    cmap="viridis",
    xticklabels=tokens,
    yticklabels=tokens
)
```

<p align="justify">

O heatmap permite visualizar o comportamento interno do mecanismo de Self-Attention.

Cada linha representa:

* o token atual.

Cada coluna representa:

* o token observado.

Quanto maior o valor:

* maior a atenção atribuída.

Isso torna interpretável um mecanismo matemático normalmente invisível em grandes modelos de linguagem.

Por exemplo, o token:

</p>

```python
"aprende"
```

<p align="justify">

pode atribuir maior atenção para:

</p>

```python
"padroes"
```

<p align="justify">

indicando que o modelo identificou relação semântica entre ação e objeto.

Da mesma forma:

</p>

```python
"artificial" → "inteligencia"
```

<p align="justify">

mostra associação contextual relevante entre os termos.

</p>

---

<p align="justify"><h1>Conclusão</h1></p>

<p align="justify">

Este projeto demonstra de forma prática e visual o funcionamento interno do mecanismo de Self-Attention, componente fundamental das arquiteturas Transformer modernas.

Ao implementar explicitamente:

* Queries;
* Keys;
* Values;
* Scores de atenção;
* Softmax;
* Combinação ponderada;

o projeto mostra entendimento real sobre o núcleo matemático utilizado em LLMs modernos.

Além disso, a utilização de heatmaps torna possível visualizar relações semânticas aprendidas pelo modelo, transformando um processo matemático abstrato em algo interpretável e intuitivo.

Esse tipo de implementação possui alto valor para portfólio técnico porque demonstra domínio de:

* Deep Learning;
* NLP;
* Transformers;
* Álgebra Linear;
* PyTorch;
* Interpretabilidade de modelos.

Mais importante ainda, o projeto evidencia capacidade de compreender não apenas o uso de modelos prontos, mas também os mecanismos internos que tornam possível o funcionamento de sistemas como GPT, BERT e LLaMA.

</p>
```

