# Implementación de Agentes de LangChain con Streamlit
Basado en el tutorial de Agentes de [LangChain](https://python.langchain.com/docs/modules/agents/quick_start)

Este repositorio permite implementar dos tipos de Agentes de LangChain: un agente sin memoria y uno con memoria.

![image](https://github.com/sergiovogt/agents-langchain-streamlit/assets/159809335/42209b13-2a27-4443-9e9e-00d898cca1a3)

## Requerimientos
- [LangChain library](https://python.langchain.com/en/latest/index.html)
- [OpenAI API key](https://platform.openai.com/)
- [Tavily API key](https://tavily.com/#api)

## Installation

#### 1. Clonar el repositorio

```bash
git clone https://github.com/sergiovogt/agents-langchain-streamlit.git
```

#### 2. Crear el entorno

``` bash
cd agents-langchain-streamlit
python -m venv env
source env/bin/activate
```

#### 3. Instalar las dependencias requeridas
``` bash
pip install -r requirements.txt
```

Primero, crear el archvio `.env` en el directorio raiz del proyecto. Dentro del archvio, agreagar la API Key de OpenAI:

```makefile
OPENAI_API_KEY="agregar_aquí_la_apikey"
```

Guardar el archivo y cerrarlo. En el script de Python, cargar el archivo `.env` usando el siguiente código (ya está cargado en [frontend.py] y en [frontend_mem.py]:
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

Ahora tu entorno de Python está listo! ya puedes continuar...

## Tutoriales
Para ver más tutoriales, podés visitar mi canal de YouTube:  [youtube.com/@sergiovogtds1998](https://youtube.com/@sergiovogtds1998)
