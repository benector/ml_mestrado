# Trabalho de Aprendizado de Máquina: Reconhecimento de ações

O projeto contém duas pastas, `kth` e `ucf`. Execute os comandos dentro da pasta do dataset correspondente para que os caminhos relativos usados pelos scripts sejam resolvidos corretamente.

## Downloads dos datasets

Baixe os vídeos nas páginas oficiais:

- **KTH:** [https://www.csc.kth.se/cvap/actions/](https://www.csc.kth.se/cvap/actions/)
- **UCF11 (YouTube Action Dataset):** [https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)

Após baixar e extrair os arquivos, ajuste os caminhos dos datasets no `feat_extract.py` de cada pasta antes de executar a extração de características.

## 1. Preparar o ambiente

Na pasta que contém o arquivo `environment.yml`, crie e ative o ambiente Conda:

```bash
conda env create -f environment.yml
conda activate reconhecimento-acoes
```

Se o ambiente já existir e o YAML tiver sido atualizado:

```bash
conda env update -n reconhecimento-acoes -f environment.yml
conda activate reconhecimento-acoes
```

Antes de extrair as características, confira no `feat_extract.py` de cada dataset os caminhos dos vídeos e de saída. Ajuste-os à organização dos dados no seu computador.

## 2. Executar os experimentos no KTH

### Extrair as características

A partir da raiz do projeto:

```bash
cd kth
python feat_extract.py
```

As características HOF serão salvas na pasta `hof_features_kth`.

### Analisar os hiperparâmetros

Após concluir a extração:

```bash
python hyper.py
```

Confira no script se o caminho de entrada aponta para as características extraídas na etapa anterior. Use os resultados da análise para definir a configuração do experimento final.

### Executar a configuração desejada

Abra `train_kth_final.py` e modifique diretamente no arquivo a configuração do modelo que deseja avaliar. Confira também os caminhos de entrada e de saída.

Depois de salvar as alterações, execute:

```bash
python train_kth_final.py
```

## 3. Executar os experimentos no UCF11

A partir da raiz do projeto:

```bash
cd ucf
```

Se você estiver dentro de `kth`, use `cd ../ucf`.

### Extrair as características

```bash
python feat_extract.py
```

As características serão salvas em um arquivo `.npz`. O nome e o caminho desse arquivo são definidos no script de extração.

### Analisar os hiperparâmetros

Após concluir a extração:

```bash
python hyper.py
```

Confira se o caminho de entrada aponta para o `.npz` gerado anteriormente. Use os resultados da análise para definir a configuração do experimento final.

### Executar a configuração desejada

Abra `train_ucf_final.py` e modifique diretamente no arquivo a configuração do modelo que deseja avaliar. Confira também os caminhos de entrada e de saída.

Depois de salvar as alterações, execute:

```bash
python train_ucf_final.py
```

> Os comandos finais seguem o padrão informado `train_[dataset]_final.py`. Se o arquivo da pasta `ucf` usar o nome `train_ucf11_final.py`, substitua o nome no comando acima.

## 4. Ordem de execução e novas avaliações

| Etapa | KTH | UCF11 |
| --- | --- | --- |
| Extração | `feat_extract.py` → `hof_features_kth` | `feat_extract.py` → arquivo `.npz` |
| Análise de hiperparâmetros | `hyper.py` | `hyper.py` |
| Avaliação final | Editar e executar `train_kth_final.py` | Editar e executar `train_ucf_final.py` |

A análise de hiperparâmetros é opcional quando a configuração desejada já estiver definida. As características podem ser reutilizadas em diferentes experimentos; não é necessário repetir a extração a cada alteração do classificador.

Escolha os hiperparâmetros pelos resultados de validação do conjunto de desenvolvimento. Preserve o teste final para a avaliação da configuração definida, sem usá-lo para orientar novos ajustes.

Os locais e nomes dos arquivos de resultados são definidos em cada script. Confira essas configurações antes de executar uma nova avaliação, especialmente para evitar sobrescrever resultados anteriores.
