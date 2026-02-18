import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0.4)

# -------------------------------
# Catálogo fixo de links internos (Dra. Tatiana Gabbi)
# -------------------------------
LINKS_INTERNOS_TATIANA = [
    {
        "titulo": "Home",
        "url": "https://tatianagabbi.com.br",
        "anchor_sugerida": "Dermatologia com foco em doenças das unhas em São Paulo"
    },
    {
        "titulo": "Sobre a Dra. Tatiana Gabbi",
        "url": "https://tatianagabbi.com.br/sobre",
        "anchor_sugerida": "saiba mais sobre a Dra. Tatiana Gabbi"
    },
    {
        "titulo": "Unhas Encravadas",
        "url": "https://tatianagabbi.com.br/unhas-encravadas",
        "anchor_sugerida": "tratamento para unhas encravadas"
    },
    {
        "titulo": "Unhas Irregulares",
        "url": "https://tatianagabbi.com.br/unhas-irregulares",
        "anchor_sugerida": "tratamento para unhas irregulares"
    },
    {
        "titulo": "Melanoníquia",
        "url": "https://tatianagabbi.com.br/melanoniquia/",
        "anchor_sugerida": "entenda o que é melanoníquia e como tratar"
    },
    {
        "titulo": "Unhas Fracas",
        "url": "https://tatianagabbi.com.br/unhas-fracas/",
        "anchor_sugerida": "tratamento para unhas fracas"
    },
    {
        "titulo": "Unhas Saudáveis",
        "url": "https://tatianagabbi.com.br/unhas-saudaveis/",
        "anchor_sugerida": "como manter unhas saudáveis"
    },
    {
        "titulo": "Detox das Unhas",
        "url": "https://tatianagabbi.com.br/detox-das-unhas/",
        "anchor_sugerida": "detox das unhas: benefícios e cuidados"
    },
    {
        "titulo": "Bases Fortalecedoras",
        "url": "https://tatianagabbi.com.br/bases-fortalecedoras/",
        "anchor_sugerida": "melhores bases fortalecedoras para unhas"
    },
    {
        "titulo": "Blog",
        "url": "https://tatianagabbi.com.br/blog",
        "anchor_sugerida": "leia mais artigos no blog da Dra. Tatiana Gabbi"
    },
    {
        "titulo": "Contato",
        "url": "https://tatianagabbi.com.br/contato",
        "anchor_sugerida": "entre em contato com a Dra. Tatiana Gabbi"
    }
]

# -------------------------------
# SERP helper + whitelist para externos (autoridades médicas)
# -------------------------------
WHITELIST_EXTERNOS_TATIANA = [
    ".gov", ".gov.br", ".edu", ".edu.br",
    "sbd.org.br", "aad.org", "who.int", "nhs.uk", "cdc.gov",
    "nih.gov", "ncbi.nlm.nih.gov", "medlineplus.gov",
    "cochranelibrary.com", "dermnetnz.org",
    "schema.org", "w3.org", "developers.google.com", "support.google.com"
]

def _usa_whitelist_tatiana(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS_TATIANA)

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
        if _usa_whitelist_tatiana(url):
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
# Função principal (Dra. Tatiana Gabbi)
# -------------------------------
def build_crew_tatiana(tema: str, palavra_chave: str):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress, para a
    Dra. Tatiana Gabbi (doenças das unhas e dermatologia).

    Estilo de saída:
    - Introdução com 1 a 2 links naturais em <p>.
    - <h2> numerados: "1. ...", "2. ..."; <h3> opcionais.
    - Parágrafos curtos (2 a 4 linhas); listas <ul><li> quando fizer sentido.
    - Pelo menos UM heading contém a palavra-chave.
    - Sem <h1> e sem imagens.
    - Mínimo 1200 palavras.
    - Linkagem: >=3 internos distribuídos (intro/corpo/conclusão) e >=1 externo de autoridade.
    - Âncoras descritivas; externos com target="_blank" rel="noopener noreferrer".
    - Conclusão sem CTA; CTA apenas na assinatura final padrão da Dra. Tatiana.
    """
    llm_local = llm

    # Monta referências e links automaticamente
    dados_concorrencia_txt = buscar_concorrentes_serpapi_texto(palavra_chave)
    serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
    links_internos = LINKS_INTERNOS_TATIANA[:]  # catálogo fixo (Tatiana)
    links_externos = selecionar_links_externos_autoritativos(serp_struct, max_links=2)

    # ==== Agentes (voz dermatologia) ====
    agente_intro = Agent(
        role="Redator de Introdução (Dermatologia)",
        goal="Escrever introdução clara e acolhedora (2 a 3 parágrafos) no tom da Dra. Tatiana, citando a palavra‑chave 1x.",
        backstory="Copywriter sênior em saúde; parágrafos curtos, linguagem acessível e responsável.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura (H2/H3) numerada",
        goal="Definir 5 a 7 H2 numerados; cobrir intenção de busca do paciente; incluir a palavra‑chave em pelo menos um heading.",
        backstory="Especialista em outline SEO para saúde; títulos informativos e específicos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento (Educação em Saúde)",
        goal="Preencher cada seção com orientação prática, sem promessas; variar semântica da keyword sem stuffing.",
        backstory="Produz conteúdo útil, com exemplos, listas e linguagem clara; sem autopromoção.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão",
        goal="Encerrar resumindo aprendizados e próximos passos práticos, sem CTA comercial.",
        backstory="Fechamentos objetivos e empáticos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Editor/Unificador de HTML",
        goal="Unir tudo em HTML único (apenas body), coerente, sem redundância, mantendo numeração.",
        backstory="Editor técnico focado em semântica limpa para WordPress.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Especialista em Linkagem (EEAT)",
        goal="Inserir links internos/externos de forma natural e distribuída, priorizando autoridade médica.",
        backstory="Foco em experiência, expertise e confiabilidade.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_contato = Agent(
        role="Responsável por Assinatura (Dra. Tatiana)",
        goal="Anexar assinatura padrão da Dra. Tatiana ao final do HTML (CTA/WhatsApp + Instagram), sem alterar o conteúdo anterior.",
        backstory="Padronização e identidade da Dra. Tatiana.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior PT-BR",
        goal="Listar melhorias objetivas em clareza, gramática, estilo, linkagem e regras SEO.",
        backstory="Revisor de saúde; corta redundâncias e mantém consistência.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as melhorias preservando estrutura e linkagem.",
        backstory="Editor/Dev de HTML limpo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2 a 3 <p>) para '{tema}' usando a palavra‑chave '{palavra_chave}' apenas 1 vez.
Estilo: acolhedor, informativo, sem jargões.
Regras:
- PT‑BR; parágrafos curtos (2 a 4 linhas).
- Sem clichês e sem promessas.
- PROIBIDO: <h1> e qualquer imagem.
- Não usar headings; apenas <p>.
- Inclua 1 link interno natural no 2º parágrafo (anchor descritiva), se compatível.
Concorrência (inspiração a NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com 2 a 3 <p> e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}':
- 5 a 7 <h2> numerados ('1. ', '2. ', ...).
- Até 2 <h3> por <h2> quando fizer sentido.
- Pelo menos UM heading deve conter a palavra‑chave '{palavra_chave}' de forma natural.
- Incluir um H2 de "Erros comuns e armadilhas" e outro de "Exemplos práticos / aplicação".
- Nunca usar <h1>. Não incluir conteúdo; só <h2>/<h3>.
Baseie a cobertura na intenção de busca do paciente e lacunas dos concorrentes:
{dados_concorrencia_txt}
""".strip(),
        expected_output="Lista hierárquica com <h2> numerados e <h3> opcionais (sem conteúdo).",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos, mantendo a numeração dos H2:
- Mínimo 1200 palavras no post completo.
- <p> curtos (2 a 4 linhas); usar <ul><li> quando listar.
- Explique: o que é, por que importa, como fazer, exemplos reais.
- Variar semântica de '{palavra_chave}' sem keyword stuffing.
- Sem autopromoção e sem CTA.
- PROIBIDO inserir imagens.
- Não inventar novos headings; usar apenas os fornecidos.
- Quando fizer sentido, inclua links internos naturais no corpo (anchors descritivas).
Concorrência (inspiração a NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com <h2> numerados, <h3> opcionais, <p> e <ul><li>.",
        agent=agente_desenvolvimento
    )

    tarefa_conclusao = Task(
        description="""
Escreva a CONCLUSÃO:
- 1 a 2 <p> resumindo aprendizados e próximos passos práticos.
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
- Garantir coerência, zero repetição e manter a NUMERAÇÃO dos <h2>.
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline, QUALQUER imagem.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body).",
        agent=agente_unificador
    )

    # Links (texto) para os agentes de linkagem
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
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão) no estilo da Dra. Tatiana.

Links internos disponíveis (use pelo menos 3, distribuídos):
{links_internos_txt}

Links externos candidatos (use >=1, se listado; com target="_blank" rel="noopener noreferrer"):
{links_externos_txt}

Regras:
- Distribuição sugerida: 1 link interno na intro, 1 a 2 no corpo, 1 na conclusão (se aplicável).
- Âncoras naturais e descritivas; nunca usar "clique aqui".
- Não linkar em headings; apenas <p> e <li>.
- Não quebrar HTML semântico; sem inline style.
- Não adicionar imagens.
Saída: HTML com linkagem aplicada.
""".strip(),
        expected_output="HTML com links internos/externos aplicados.",
        agent=agente_linkagem
    )

    # Assinatura/CTA da Dra. Tatiana (NOVA VERSÃO)
    tarefa_contato = Task(
        description="""
Anexar ao FINAL do HTML a assinatura padrão ATUALIZADA da Dra. Tatiana (sem alterar o conteúdo anterior):
<p><strong><a href="https://api.whatsapp.com/send?phone=5511991578420&amp;text=Oi!%20Encontrei%20seu%20contato%20no%20site%20e%20gostaria%20de%20mais%20informações">Agende sua consulta via WhatsApp para avaliação especializada</a></strong></p>
<p><strong>Siga o Instagram da Dra. Tatiana Gabbi:</strong> <a href="https://www.instagram.com/dratatianagabbi/" target="_blank" rel="noopener noreferrer">@dratatianagabbi</a> e acompanhe conteúdos exclusivos sobre a saúde das unhas e cuidados dermatológicos ao longo do ano.</p>
<p><strong>Dra. Tatiana Gabbi</strong> - Médica Dermatologista especializada em doenças das unhas, atuando em São Paulo com excelência e cuidado personalizado</p>
""".strip(),
        expected_output="HTML final com assinatura ATUALIZADA adicionada.",
        agent=agente_contato
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT‑BR; clareza; tom acolhedor e profissional.
- Estilo: H2 numerados, parágrafos curtos, listas quando úteis, distribuição de links.
- Coerência e distribuição de links; âncoras descritivas; ausência de overstuffing de '{palavra_chave}'.
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
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only).",
        agent=agente_executor
    )

    # ==== Crew ====
    crew_tatiana = Crew(
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
    return crew_tatiana