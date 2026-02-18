import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0.4)

# -------------------------------
# Catálogo fixo de links internos (Dr. Gerson Righetto Junior)
# -------------------------------
LINKS_INTERNOS_RIGHETTO = [
    {"titulo": "Início", "url": "https://gersonrighetto.com.br/",
     "anchor_sugerida": "conheça a clínica do Dr. Gerson Righetto"},
    {"titulo": "Sobre o Dr. Gerson Righetto", "url": "https://gersonrighetto.com.br/dr-gerson-righetto/",
     "anchor_sugerida": "saiba mais sobre formação e experiência do Dr. Gerson Righetto"},
    {"titulo": "Tratamentos", "url": "https://gersonrighetto.com.br/tratamentos/",
     "anchor_sugerida": "tratamentos personalizados em ginecologia e nutrologia"},
    {"titulo": "Blog", "url": "https://gersonrighetto.com.br/blog/",
     "anchor_sugerida": "conteúdos sobre saúde íntima feminina e bem-estar"},
    {"titulo": "Contato", "url": "https://gersonrighetto.com.br/contato/",
     "anchor_sugerida": "entre em contato com a equipe do Dr. Gerson Righetto"},
]

# -------------------------------
# SERP helper + whitelist para externos (saúde da mulher / ginecologia)
# -------------------------------
WHITELIST_EXTERNOS = [
    ".gov", ".gov.br", ".edu", ".edu.br",
    "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov",
    "ms.gov.br", "saude.gov.br", "saude.pr.gov.br",
    "sbd.org.br", "febrasgo.org.br", "sbem.org.br", "ans.gov.br", "inca.gov.br",
    "jamanetwork.com", "nejm.org", "bmj.com",
    "developers.google.com", "support.google.com", "search.google.com",
    "schema.org", "w3.org",
    "moz.com", "ahrefs.com", "semrush.com", "data.gov"
]

def _usa_whitelist(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS)

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
        if _usa_whitelist(url):
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
# Função principal (Dr. Gerson Righetto)
# -------------------------------
def build_crew_gerson(tema: str, palavra_chave: str):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress, no estilo do Dr. Gerson Righetto Junior.

    Estilo de saída:
    - Introdução com 1–2 links naturais em <p>.
    - <h2> numerados: "1. ...", "2. ..."; <h3> opcionais.
    - Parágrafos curtos (2–4 linhas); listas <ul><li> quando fizer sentido.
    - Pelo menos UM heading contém a palavra-chave.
    - Sem <h1> e sem imagens.
    - Mínimo 1200 palavras.
    - Linkagem: >=3 internos distribuídos (intro/corpo/conclusão) e >=1 externo (se houver whitelist).
    - Anchors descritivas; externos com target="_blank" rel="noopener noreferrer".
    - Conclusão sem CTA; CTA na assinatura ao final.
    - Tom: médico, acolhedor e baseado em evidências; foco em saúde íntima feminina, prevenção, diagnóstico precoce, tecnologias como Laser Íntimo CO₂ quando pertinente e abordagem integrativa (ginecologia + nutrologia).
    """
    llm_local = llm

    # Monta referências e links automaticamente
    dados_concorrencia_txt = buscar_concorrentes_serpapi_texto(palavra_chave)
    serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
    links_internos = LINKS_INTERNOS_RIGHETTO[:]  # catálogo fixo
    links_externos = selecionar_links_externos_autoritativos(serp_struct, max_links=2)

    # ==== Agentes ====
    agente_intro = Agent(
        role="Redator de Introdução",
        goal="Escrever introdução clara, empática e precisa (2–3 parágrafos), citando a palavra-chave 1x.",
        backstory="Especialista em ginecologia e saúde da mulher; evita alarmismo; linguagem acessível.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura (H2/H3) com numeração",
        goal="Definir 5–7 H2 numerados; cobrir intenção de busca; incluir a palavra-chave em pelo menos um heading.",
        backstory="Outline SEO para saúde feminina, com foco em prevenção, diagnóstico e tratamento.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento",
        goal="Preencher cada seção com <p> curtos e listas úteis, variar semântica da keyword sem stuffing e sem inserir imagens.",
        backstory="Explica sinais de alerta, exames (ex.: colposcopia/USG quando aplicável), nutrologia como apoio, terapias como Laser Íntimo CO₂ quando pertinente.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão (sem CTA)",
        goal="Encerrar resumindo aprendizados e próximos passos práticos, sem convite comercial.",
        backstory="Fechamentos naturais, objetivos e acolhedores.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Unificador de Conteúdo HTML",
        goal="Unir tudo em HTML único (apenas body), coerente, sem redundância, com numeração dos H2 e sem imagens.",
        backstory="Editor focado em semântica, acessibilidade e limpeza de HTML.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Planejador e Implementador de Linkagem",
        goal="Inserir links internos/externos de forma natural e distribuída, respeitando todas as regras.",
        backstory="Especialista em internal linking e EEAT em saúde da mulher.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_contato = Agent(
        role="Responsável por Contato e Assinatura",
        goal="Anexar assinatura institucional do Dr. Gerson Righetto ao final do HTML (CTA/WhatsApp), sem alterar o conteúdo anterior.",
        backstory="Padronização e identidade médica.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior",
        goal="Listar melhorias objetivas (bullets) em clareza, gramática, tom médico, distribuição de links e regras SEO.",
        backstory="Revisor PT-BR para conteúdos médicos; elimina ambiguidades e redundâncias.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as melhorias preservando estrutura semântica e linkagem.",
        backstory="Editor/Dev de HTML limpo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2–3 <p>) para '{tema}' usando a palavra-chave '{palavra_chave}' apenas 1 vez.
Tom médico, acolhedor e baseado em evidências.
Regras:
- PT-BR; parágrafos curtos (2–4 linhas).
- Sem clichês ou promessas; linguagem clara e empática.
- PROIBIDO: <h1> e qualquer imagem.
- Não usar headings na introdução; só <p>.
- Se houver âncora compatível, inclua 1 link interno natural no 2º parágrafo (anchor descritiva).
Concorrência (inspiração – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com 2–3 <p> (sem imagens) e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}':
- 5–7 <h2> numerados com prefixo '1. ', '2. ', '3. ' ...
- Até 2 <h3> por <h2> quando fizer sentido (sem numeração).
- Pelo menos UM heading (<h2> ou <h3>) deve conter a palavra-chave '{palavra_chave}' de forma natural.
- Incluir um H2 equivalente a "Erros comuns e armadilhas" e outro a "Exemplos práticos / aplicação".
- Títulos específicos, claros e orientados a decisão da paciente.
- Nunca usar <h1>. Não incluir conteúdo; só <h2>/<h3>.
Baseie a cobertura na intenção de busca e em lacunas/oportunidades dos concorrentes:
{dados_concorrencia_txt}
""".strip(),
        expected_output="Lista hierárquica com <h2> numerados e <h3> opcionais (sem conteúdo).",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos, mantendo a numeração dos H2:
- Mínimo de 1200 palavras no post completo (será validado no unificador).
- <p> curtos (2–4 linhas); usar <ul><li> quando listar.
- Explicar: o que é, sinais/queixas comuns, prevenção, diagnóstico (ex.: exames ginecológicos), tratamentos (ex.: Laser Íntimo CO₂ quando pertinente), nutrologia como apoio e acompanhamento contínuo.
- Variar semântica de '{palavra_chave}' sem stuffing.
- Sem autopromoção e sem CTA.
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
- 1–2 <p> resumindo aprendizados e próximos passos práticos (quando procurar ginecologista, sinais de alerta, rotina preventiva).
- Zero CTA (o CTA fica na assinatura).
- Inclua 1 link interno natural se ainda não houver link na conclusão.
- Não inserir imagens.
""".strip(),
        expected_output="Conclusão em <p>, possivelmente com 1 link interno.",
        agent=agente_conclusao
    )

    tarefa_unificar = Task(
        description="""
Una introdução, corpo e conclusão em um único HTML (conteúdo do body, sem <body>).
Regras:
- Garantir coerência, zero repetição e manter a NUMERAÇÃO dos <h2> (1., 2., 3., ...).
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline, QUALQUER tag de imagem.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body, sem imagens).",
        agent=agente_unificador
    )

    # Links disponíveis para a etapa de linkagem
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
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão).

Links internos disponíveis (use pelo menos 3, distribuídos):
{links_internos_txt}

Links externos candidatos (use >=1, se listado; com target="_blank" rel="noopener noreferrer"):
{links_externos_txt}

Regras:
- Distribuição sugerida: 1 link interno na intro, 1–2 no corpo, 1 na conclusão (se aplicável).
- Âncoras naturais e descritivas; nunca usar "clique aqui".
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

<p><strong>Clique aqui para agendar uma avaliação pelo WhatsApp</strong></p>
<p><a href="https://api.whatsapp.com/send?phone=5541999380202&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank" rel="noopener noreferrer">https://api.whatsapp.com/send?phone=5541999380202</a></p>
<p><strong>Dr. Gerson Righetto Junior</strong><br>Ginecologista e Obstetra<br>Av. Sete de Setembro, 4214 - cj 1707 - Batel, Curitiba – PR</p>
""".strip(),
        expected_output="HTML final com assinatura adicionada.",
        agent=agente_contato
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT-BR; clareza; tom médico (preciso, acolhedor e baseado em evidências).
- Estilo: H2 numerados, parágrafos curtos, listas quando úteis, distribuição de links.
- Coerência e distribuição de links; âncoras descritivas; ausência de overstuffing de '{palavra_chave}'.
- Respeito às proibições de imagens e de <h1>.
Saída: lista de melhorias acionáveis em bullets JSON-like:
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
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only, sem imagens).",
        agent=agente_executor
    )

    # ==== Crew ====
    crew_righetto = Crew(
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
    return crew_righetto
