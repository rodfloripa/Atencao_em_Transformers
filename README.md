
# Visualizando Self-Attention em Transformers

<div align="justify">

Este projeto implementa uma versão educacional do mecanismo de <b>Self-Attention</b>, núcleo matemático utilizado em arquiteturas modernas como GPT, BERT, LLaMA e Gemini.

O objetivo não é apenas utilizar um Transformer pronto, mas sim mostrar detalhadamente:

<ul>
<li>como os embeddings são criados</li>
<li>como Queries, Keys e Values funcionam</li>
<li>como a matriz de atenção é calculada</li>
<li>como os pesos de atenção são distribuídos</li>
<li>como o modelo aprende relações semânticas entre palavras</li>
</ul>

Além disso, o projeto gera um <b>heatmap interpretável</b>, permitindo visualizar para quais tokens cada palavra está “olhando”.

Esse tipo de projeto possui excelente valor para portfólio porque demonstra domínio de:

<ul>
<li>Deep Learning</li>
<li>NLP</li>
<li>Transformers</li>
<li>Álgebra Linear</li>
<li>PyTorch</li>
<li>Interpretabilidade de modelos</li>
</ul>

</div>

---

# Funcionamento Geral

<div align="justify">

O pipeline do projeto segue exatamente o fluxo interno utilizado em Transformers reais.

A execução ocorre nas seguintes etapas:

<ol>
<li>Tokenização da frase</li>
<li>Conversão para IDs numéricos</li>
<li>Criação dos embeddings</li>
<li>Projeção em:
<ul>
<li>Query (Q)</li>
<li>Key (K)</li>
<li>Value (V)</li>
</ul>
</li>
<li>Cálculo dos scores de atenção</li>
<li>Aplicação do Softmax</li>
<li>Geração da saída ponderada</li>
<li>Visualização do mapa de atenção</li>
</ol>

</div>

---

# Fórmula Principal da Self-Attention

```math
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
````

<div align="justify">

Essa fórmula representa o mecanismo central dos Transformers.

Ela mede a similaridade entre Queries e Keys utilizando produto escalar.

O resultado é normalizado pelo Softmax para gerar probabilidades de atenção.

Depois, essas probabilidades ponderam os Values, produzindo a saída final contextualizada.

</div>

---

# 1. Tokenização

```python
sentence = "a inteligencia artificial aprende padroes"

tokens = sentence.split()
```

<div align="justify">

Aqui a frase é dividida em palavras individuais.

Cada palavra será tratada como um token independente pelo Transformer.

Exemplo:

</div>

```python
['a', 'inteligencia', 'artificial', 'aprende', 'padroes']
```

<div align="justify">

Em modelos reais, a tokenização costuma ser mais sofisticada, utilizando técnicas como:

<ul>
<li>Byte Pair Encoding (BPE)</li>
<li>SentencePiece</li>
<li>WordPiece</li>
</ul>

Neste projeto, o split simples facilita a interpretação visual do funcionamento interno do modelo.

</div>

---

# 2. Criação do Vocabulário

```python
vocab = {word: idx for idx, word in enumerate(tokens)}

token_ids = torch.tensor([vocab[t] for t in tokens])
```

<div align="justify">

Transformers não trabalham diretamente com texto.

Cada palavra precisa ser convertida em um índice numérico.

Exemplo:

</div>

```python
{
 'a': 0,
 'inteligencia': 1,
 'artificial': 2,
 'aprende': 3,
 'padroes': 4
}
```

<div align="justify">

Esses IDs serão usados pela camada de embedding.

</div>

---

# 3. Embeddings

```python
embedding = nn.Embedding(len(vocab), embedding_dim)

x = embedding(token_ids)
```

<div align="justify">

Embeddings transformam tokens em vetores densos de números reais.

Esses vetores representam características semânticas das palavras.

Exemplo:

</div>

```python
'inteligencia' → [0.12, -0.45, 0.88, ...]
```

<div align="justify">

Modelos reais utilizam embeddings com centenas ou milhares de dimensões.

Neste projeto utilizamos dimensão reduzida para facilitar visualização e entendimento.

</div>

---

# 4. Classe SelfAttention

```python
class SelfAttention(nn.Module):
```

<div align="justify">

Essa classe implementa o núcleo matemático do Transformer.

Ela cria três projeções lineares:

<ul>
<li>Query</li>
<li>Key</li>
<li>Value</li>
</ul>

</div>

```python
self.query = nn.Linear(embed_dim, embed_dim)
self.key = nn.Linear(embed_dim, embed_dim)
self.value = nn.Linear(embed_dim, embed_dim)
```

<div align="justify">

Essas camadas aprendem representações diferentes do mesmo token.

</div>

---

# 5. Geração de Q, K e V

```python
Q = self.query(x)
K = self.key(x)
V = self.value(x)
```

<div align="justify">

Cada embedding é projetado em três espaços diferentes:

<ul>
<li>Query → o que o token procura</li>
<li>Key → o que o token oferece</li>
<li>Value → informação carregada</li>
</ul>

O Transformer compara Queries com Keys para medir relevância contextual.

</div>

---

# 6. Attention Scores

```python
scores = torch.matmul(Q, K.T) / np.sqrt(dk)
```

<div align="justify">

Aqui ocorre o cálculo principal da atenção.

O produto escalar mede similaridade entre tokens.

Quanto maior o score:

<ul>
<li>maior a relação semântica</li>
<li>maior será a atenção atribuída</li>
</ul>

A divisão por raiz quadrada de d_k evita crescimento excessivo dos valores, estabilizando o treinamento.

</div>

```math
scores=\frac{QK^T}{\sqrt{d_k}}
```

---

# 7. Softmax

```python
attention_weights = torch.softmax(scores, dim=-1)
```

<div align="justify">

O Softmax transforma os scores em probabilidades.

Cada linha da matriz passa a somar 1.

Isso permite interpretar os valores como níveis de atenção entre palavras.

</div>

```math
AttentionWeights=softmax(scores)
```

---

# 8. Saída Final

```python
output = torch.matmul(attention_weights, V)
```

<div align="justify">

Os pesos de atenção são usados para combinar os Values.

Isso gera uma nova representação contextualizada dos tokens.

Em outras palavras:

cada palavra passa a carregar informações relevantes das demais palavras da frase.

</div>

```math
Output=AttentionWeights\times V
```

---

# 9. Heatmap de Atenção

```python
sns.heatmap(
    weights_np,
    annot=True,
    cmap="viridis",
    xticklabels=tokens,
    yticklabels=tokens
)
```

<div align="justify">

O heatmap permite visualizar o comportamento interno do mecanismo de Self-Attention.

Cada linha representa:

<ul>
<li>o token atual</li>
</ul>

Cada coluna representa:

<ul>
<li>o token observado</li>
</ul>

Quanto maior o valor:

<ul>
<li>maior a atenção atribuída</li>
</ul>

Isso torna interpretável um mecanismo matemático normalmente invisível em grandes modelos de linguagem.

Por exemplo, o token:

</div>

```python
"aprende"
```

<div align="justify">

pode atribuir maior atenção para:

</div>

```python
"padroes"
```

<div align="justify">

indicando que o modelo identificou relação semântica entre ação e objeto.

Da mesma forma:

</div>

```python
"artificial" → "inteligencia"
```

<div align="justify">

mostra associação contextual relevante entre os termos.

</div>

---

# Conclusão

<div align="justify">

Este projeto demonstra de forma prática e visual o funcionamento interno do mecanismo de Self-Attention, componente fundamental das arquiteturas Transformer modernas.

Ao implementar explicitamente:

<ul>
<li>Queries</li>
<li>Keys</li>
<li>Values</li>
<li>Scores de atenção</li>
<li>Softmax</li>
<li>Combinação ponderada</li>
</ul>

o projeto mostra entendimento real sobre o núcleo matemático utilizado em LLMs modernos.

Além disso, a utilização de heatmaps torna possível visualizar relações semânticas aprendidas pelo modelo, transformando um processo matemático abstrato em algo interpretável e intuitivo.

Esse tipo de implementação possui alto valor para portfólio técnico porque demonstra domínio de:

<ul>
<li>Deep Learning</li>
<li>NLP</li>
<li>Transformers</li>
<li>Álgebra Linear</li>
<li>PyTorch</li>
<li>Interpretabilidade de modelos</li>
</ul>

Mais importante ainda, o projeto evidencia capacidade de compreender não apenas o uso de modelos prontos, mas também os mecanismos internos que tornam possível o funcionamento de sistemas como GPT, BERT e LLaMA.

</div>
```

