import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0.4)

# -------------------------------
# Catálogo fixo de links internos (Dra. Angélica Bauer)
# -------------------------------
LINKS_INTERNOS_ANGELICA = [
    {
        "titulo": "Home",
        "url": "https://www.angelicabauer.com.br/",
        "anchor_sugerida": "Dermatologia com foco em alopecia em Porto Alegre"
    },
    {
        "titulo": "Tratamento Dermatológico Clínico",
        "url": "https://www.angelicabauer.com.br/tratamento-dermatologico-clinico/",
        "anchor_sugerida": "tratamentos dermatológicos clínicos com a Dra. Angélica Bauer"
    },
    {
        "titulo": "Tricologia",
        "url": "https://www.angelicabauer.com.br/tricologia/",
        "anchor_sugerida": "cuidados e tratamentos em tricologia"
    },
    {
        "titulo": "Mapeamento Corporal",
        "url": "https://www.angelicabauer.com.br/mapeamento-corporal/",
        "anchor_sugerida": "mapeamento corporal para prevenção e diagnóstico precoce"
    },
    {
        "titulo": "Cirurgia Dermatológica",
        "url": "https://www.angelicabauer.com.br/cirurgia-dermatologica/",
        "anchor_sugerida": "procedimentos de cirurgia dermatológica"
    },
    {
        "titulo": "Procedimentos Estéticos e Tecnologias",
        "url": "https://www.angelicabauer.com.br/procedimentos-esteticos-e-tecnologias/",
        "anchor_sugerida": "procedimentos estéticos e tecnologias para saúde da pele"
    },
    {
        "titulo": "Blog",
        "url": "https://www.angelicabauer.com.br/blog/",
        "anchor_sugerida": "conteúdos sobre saúde da pele e cabelos no blog da Dra. Angélica"
    }
]

# -------------------------------
# SERP helper + whitelist para externos (autoridades médicas)
# -------------------------------
WHITELIST_EXTERNOS_ANGELICA = [
    ".gov", ".gov.br", ".edu", ".edu.br",
    "sbd.org.br", "aad.org", "who.int", "nhs.uk", "cdc.gov",
    "nih.gov", "ncbi.nlm.nih.gov", "medlineplus.gov",
    "cochranelibrary.com", "dermnetnz.org",
    "schema.org", "w3.org", "developers.google.com", "support.google.com"
]

def _usa_whitelist_angelica(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS_ANGELICA)

def buscar_concorrentes_serpapi_struct(palavra_chave: str) -> list[dict]:
    search = GoogleSearch({
        "q": palavra_chave,
        "hl": "pt-br",
        "gl": "br",
        "num": 10,
        "api_key": os.getenv("SERPAPI_API_KEY")
    })
    d = search.get_dict()
    return d.get("organic_results", []) or []

def selecionar_links_externos_autoritativos(resultados_serp: list[dict], max_links: int = 2) -> list[dict]:
    candidatos, vistos = [], set()
    for r in resultados_serp:
        url = r.get("link") or r.get("url") or ""
        titulo = (r.get("title") or "").strip()
        if not url or url in vistos:
            continue
        if _usa_whitelist_angelica(url):
            candidatos.append({
                "titulo": titulo[:90] or "Fonte externa",
                "url": url,
                "anchor_sugerida": (titulo[:70].lower() or "fonte oficial")
            })
            vistos.add(url)
        if len(candidatos) >= max_links:
            break
    return candidatos

def buscar_concorrentes_serpapi_texto(palavra_chave: str) -> str:
    """Versão textual só para inspiração (NÃO copiar)."""
    results = buscar_concorrentes_serpapi_struct(palavra_chave)
    output = []
    for res in results:
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "") or res.get("url", "")
        output.append(f"Título: {titulo}\nTrecho: {snippet}\nURL: {link}\n")
    return "\n".join(output)

# -------------------------------
# Função principal (Dra. Angélica Bauer)
# -------------------------------
def build_crew_angelica(tema: str, palavra_chave: str):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress.

    Estilo de saída:
    - Introdução com 1–2 links naturais em <p>.
    - <h2> numerados: "1. ...", "2. ..."; <h3> opcionais.
    - Parágrafos curtos (2–4 linhas); listas <ul><li> quando fizer sentido.
    - Pelo menos UM heading contém a palavra‑chave.
    - Sem <h1> e sem imagens.
    - Mínimo 1200 palavras.
    - Linkagem: >=3 internos distribuídos (intro/corpo/conclusão) e >=1 externo (se houver whitelist).
    - Anchors descritivas; externos com target="_blank" rel="noopener noreferrer".
    - Conclusão sem CTA comercial; CTA na assinatura ao final.
    - Foco exclusivo em dermatologia e tricologia, especialmente alopecia.
    """
    llm_local = llm

    # Monta referências e links automaticamente
    dados_concorrencia_txt = buscar_concorrentes_serpapi_texto(palavra_chave)
    serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
    links_internos = LINKS_INTERNOS_ANGELICA[:]  # catálogo fixo
    links_externos = selecionar_links_externos_autoritativos(serp_struct, max_links=2)

    # ==== Agentes ====
    agente_intro = Agent(
        role="Redator de Introdução Médica",
        goal="Escrever introdução clara, acolhedora e empática (2–3 parágrafos) no estilo médico educativo, citando a palavra‑chave 1x e focando em dermatologia/tricologia.",
        backstory="Copywriter especializado em comunicação médica; evita promessas exageradas; tom acolhedor mas científico; parágrafos curtos; sem imagens.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura Médica (H2/H3) com numeração",
        goal="Definir 5–7 H2 numerados (1., 2., 3., ...), com H3 opcionais; cobrir aspectos dermatológicos/tricológicos e incluir a palavra‑chave em pelo menos um heading.",
        backstory="Especialista em estrutura de conteúdo médico; nunca usa H1; títulos específicos focados em saúde da pele e cabelos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento Médico",
        goal="Preencher cada seção com <p> curtos, informações técnicas precisas, listas úteis, variar semântica da keyword sem stuffing e sem inserir imagens.",
        backstory="Conteúdo médico educativo, científico mas acessível, com foco em dermatologia e tricologia, sem promessas exageradas.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão Médica (sem CTA)",
        goal="Encerrar resumindo aprendizados médicos e orientações práticas sem convite comercial direto.",
        backstory="Fechamentos educativos focados em saúde e bem-estar do paciente.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Unificador de Conteúdo HTML Médico",
        goal="Unir tudo em HTML único (apenas body), coerente, sem redundância, com numeração dos H2 e sem imagens, mantendo rigor científico.",
        backstory="Editor técnico focado em conteúdo médico, semântica e limpeza de HTML.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Planejador e Implementador de Linkagem Médica",
        goal="Inserir links internos/externos de forma natural e distribuída, priorizando fontes médicas autoritativas e serviços dermatológicos.",
        backstory="Especialista em EEAT médico e linkagem para autoridade em dermatologia.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_contato = Agent(
        role="Responsável por Contato e Assinatura Médica",
        goal="Anexar assinatura institucional da Dra. Angélica ao final do HTML (CTA/WhatsApp), sem alterar o conteúdo anterior.",
        backstory="Padronização e identidade institucional médica.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior Médico",
        goal="Listar melhorias objetivas (bullets) em clareza, precisão médica, gramática, distribuição de links e adequação ao público leigo.",
        backstory="Revisor PT‑BR especializado em textos médicos; corta redundâncias; mantém rigor científico acessível.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões Médicas",
        goal="Aplicar todas as melhorias preservando estrutura semântica, linkagem e precisão médica.",
        backstory="Editor/Dev de HTML limpo especializado em conteúdo médico.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2–3 <p>) para '{tema}' usando a palavra‑chave '{palavra_chave}' apenas 1 vez.
Estilo médico educativo com tom acolhedor mas científico.
Regras:
- PT‑BR; parágrafos curtos (2–4 linhas).
- Evitar promessas exageradas e linguagem sensacionalista.
- Foco em dermatologia/tricologia, especialmente questões relacionadas à alopecia e saúde capilar.
- PROIBIDO: <h1> e qualquer imagem.
- Não usar headings na introdução; só <p>.
- Se houver âncora compatível, inclua 1 link interno natural no 2º parágrafo (anchor descritiva).
- Tom empático mas profissional, adequado para pacientes buscando informação médica.
Concorrência (inspiração – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com 2–3 <p> (sem imagens) e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}' no estilo médico educativo:
- 5–7 <h2> numerados com prefixo '1. ', '2. ', '3. ' ...
- Até 2 <h3> por <h2> quando fizer sentido (sem numeração).
- Pelo menos UM heading (<h2> ou <h3>) deve conter a palavra‑chave '{palavra_chave}' de forma natural.
- Incluir um H2 sobre "Quando procurar um dermatologista" e outro sobre "Cuidados e prevenção".
- Títulos específicos, claros e focados em aspectos dermatológicos/tricológicos.
- Nunca usar <h1>. Não incluir conteúdo; só <h2>/<h3>.
- Abordar causas, sintomas, tratamentos e prevenção quando aplicável.
Baseie a cobertura na intenção de busca médica e em lacunas/oportunidades dos concorrentes:
{dados_concorrencia_txt}
""".strip(),
        expected_output="Lista hierárquica com <h2> numerados e <h3> opcionais (sem conteúdo).",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos, mantendo a numeração dos H2:
- Mínimo de 1200 palavras no post completo (será validado no unificador).
- <p> curtos (2–4 linhas); usar <ul><li> quando listar sintomas, tratamentos ou cuidados.
- Explicar: o que é, causas, sintomas, como tratar, prevenção, quando procurar ajuda médica.
- Variar semântica de '{palavra_chave}' sem stuffing.
- Manter rigor científico mas linguagem acessível ao público leigo.
- Foco em dermatologia e tricologia, especialmente alopecia e saúde capilar.
- Sem promessas exageradas e sem CTA comercial no desenvolvimento.
- PROIBIDO inserir imagens.
- Não inventar novos headings; usar apenas os fornecidos.
- Quando fizer sentido, inclua links internos naturais no corpo (anchors descritivas).
Concorrência (inspiração – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com <h2> numerados, <h3> opcionais, <p> e <ul><li> (sem imagens).",
        agent=agente_desenvolvimento
    )

    tarefa_conclusao = Task(
        description="""
Escreva a CONCLUSÃO:
- 1–2 <p> resumindo os principais pontos médicos e orientações para o cuidado da saúde capilar/dermatológica.
- Zero CTA comercial (o CTA fica na assinatura).
- Inclua 1 link interno natural se ainda não houver link na conclusão.
- Tom profissional e acolhedor, reforçando a importância do acompanhamento médico especializado.
- Não inserir imagens.
""".strip(),
        expected_output="Conclusão em <p>, possivelmente com 1 link interno.",
        agent=agente_conclusao
    )

    tarefa_unificar = Task(
        description="""
Una introdução, corpo e conclusão em um único HTML (conteúdo do body, sem <body>).
Regras:
- Garantir coerência médica, zero repetição e manter a NUMERAÇÃO dos <h2> (1., 2., 3., ...).
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline, QUALQUER tag de imagem.
- Manter rigor científico e linguagem acessível.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body, sem imagens).",
        agent=agente_unificador
    )

    # Links colados na descrição
    links_internos_txt = "\n".join(
        f"- {li['titulo']}: {li['url']} | âncora sugerida: {li['anchor_sugerida']}"
        for li in links_internos
    )
    links_externos_txt = "\n".join(
        f"- {le['titulo']}: {le['url']} | âncora sugerida: {le['anchor_sugerida']}"
        for le in links_externos
    ) or "(nenhum externo autorizado encontrado)"

    tarefa_linkagem = Task(
        description=f"""
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão) seguindo padrões médicos de qualidade.

Links internos disponíveis (use pelo menos 3, distribuídos):
{links_internos_txt}

Links externos candidatos (use >=1, se listado; com target="_blank" rel="noopener noreferrer"):
{links_externos_txt}

Regras:
- Distribuição sugerida: 1 link interno na intro, 1–2 no corpo, 1 na conclusão (se aplicável).
- Priorizar links para serviços relacionados ao tema (tricologia, tratamentos dermatológicos, etc.).
- Âncoras naturais e descritivas; nunca usar "clique aqui".
- Links externos apenas para fontes médicas autoritativas (SBD, estudos, etc.).
- Não linkar em headings; apenas <p> e <li>.
- Não quebrar HTML semântico; sem inline style.
- Não adicionar imagens.
Saída: HTML com linkagem aplicada.
""".strip(),
        expected_output="HTML com links internos/externos aplicados (sem imagens).",
        agent=agente_linkagem
    )

    tarefa_contato = Task(
        description="""
Anexar ao FINAL do HTML a assinatura institucional (sem alterar o conteúdo anterior):
<p>Agende sua consulta e aprenda como manter sua pele jovem, firme e luminosa após os 40!<br>
<a href="https://api.whatsapp.com/send?phone=5551999216941&text=Olá,%20estava%20navegando%20pelo%20site%20e%20gostaria%20de%20mais%20informações" target="_blank" rel="noopener noreferrer">Entre em contato pelo WhatsApp</a></p>
<p><strong>Dra Angélica Bauer</strong><br>Dermatologista em Porto Alegre</p>
""".strip(),
        expected_output="HTML final com assinatura adicionada.",
        agent=agente_contato
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT‑BR; clareza; precisão médica; tom profissional mas acessível.
- Estrutura médica educativa: H2 numerados, parágrafos curtos, listas quando úteis, distribuição de links.
- Coerência científica e distribuição de links; âncoras descritivas; ausência de overstuffing de '{palavra_chave}'.
- Adequação ao público leigo interessado em saúde dermatológica/capilar.
- Respeito às proibições de imagens e de <h1>.
Saída: lista de melhorias acionáveis em bullets JSON‑like:
- {{"campo":"trecho/resumo","problema":"...","acao":"..."}}
""".strip(),
        expected_output="Bullets com melhorias acionáveis.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="""
Aplique TODAS as melhorias propostas, preservando:
- Estrutura semântica (<h2> numerados/<h3>/<p>/<ul><li>/<a>).
- Linkagem já aplicada (ajuste âncora só se necessário).
- Ausência de imagens e de <h1>.
- Rigor científico e linguagem acessível.
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only, sem imagens).",
        agent=agente_executor
    )

    # ==== Crew ====
    crew_angelica = Crew(
        agents=[
            agente_intro, agente_outline, agente_desenvolvimento, agente_conclusao,
            agente_unificador, agente_linkagem, agente_contato,
            agente_revisor, agente_executor
        ],
        tasks=[
            tarefa_intro, tarefa_outline, tarefa_desenvolvimento, tarefa_conclusao,
            tarefa_unificar, tarefa_linkagem, tarefa_contato,
            tarefa_revisar, tarefa_corrigir
        ],
        verbose=True
    )
    return crew_angelica