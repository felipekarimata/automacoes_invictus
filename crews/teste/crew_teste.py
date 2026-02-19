import os
from dotenv import load_dotenv
from serpapi.google_search import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI


load_dotenv()
llm = ChatOpenAI(temperature=0.4)

def buscar_concorrentes_serpapi(palavra_chave):
    search = GoogleSearch({
        "q": palavra_chave,
        "hl": "pt-br",
        "gl": "br",
        "num": 5,
        "api_key": os.getenv("SERPAPI_API_KEY")
    })
    results = search.get_dict()
    output = []
    for res in results.get("organic_results", []):
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "")
        output.append(f"Título: {titulo}\nTrecho: {snippet}\nURL: {link}\n")
    return "\n".join(output)

def build_crew_invictus_conteudo(tema: str, palavra_chave: str, links_internos: list[dict], links_externos: list[dict] | None = None):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress.
    - Sem <html>, <head>, <title>, meta description ou estilos inline.
    - Inclui linkagem interna/externa distribuída (intro/corpo/conclusão).
    - Conclusão sem CTA (CTA fica apenas na assinatura institucional).
    - Mínimo 1200 palavras.

    links_internos: [{"titulo": "...", "url": "...", "anchor_sugerida": "..."}]
    links_externos: [{"titulo": "...", "url": "...", "anchor_sugerida": "..."}]
    """
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)
    llm_local = llm

    # ==== Agentes ====
    agente_intro = Agent(
        role="Redator de Introdução",
        goal="Escrever introdução clara e persuasiva com 2–3 parágrafos, contextualizando o tema e citando a palavra‑chave 1x.",
        backstory="Copywriter sênior B2B; evita clichês, promete só o que entrega; parágrafos curtos.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura (H2/H3)",
        goal="Definir 5–7 H2 e H3 opcionais, cobrindo integralmente a intenção de busca.",
        backstory="Especialista em outline orientado a SEO e escaneabilidade.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento",
        goal="Desenvolver cada H2/H3 com <p> curtos e <ul><li> práticos, variando semântica da keyword sem overstuffing.",
        backstory="Criador de conteúdo útil, direto e aplicável.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão (sem CTA)",
        goal="Encerrar resumindo aprendizados e próximos passos práticos, sem convite comercial.",
        backstory="Especialista em fechamentos naturais.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Unificador de Conteúdo HTML",
        goal="Unir as partes em um único HTML (apenas body), coerente e sem redundância.",
        backstory="Editor técnico com foco em semântica e limpeza de HTML.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Planejador e Implementador de Linkagem",
        goal="Inserir links internos/externos de forma natural e distribuída.",
        backstory="Especialista em internal linking e EEAT.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_contato = Agent(
        role="Responsável por Contato e Assinatura",
        goal="Anexar assinatura institucional da Invictus ao final do HTML.",
        backstory="Garante padronização e identidade institucional.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior",
        goal="Apontar melhorias objetivas (itens acionáveis) em clareza, gramática e tom.",
        backstory="Revisor PT‑BR; corta gordura, mantém utilidade.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar integralmente as melhorias mantendo a estrutura semântica e links.",
        backstory="Editor/Dev de HTML limpo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2–3 <p>) para '{tema}' usando a palavra‑chave '{palavra_chave}' apenas 1 vez.
Regras:
- PT‑BR; parágrafos curtos (2–4 linhas).
- Explique o problema/dor e a relevância prática.
- Sem clichês, sem promessas vazias.
- Não usar <h1>. Só <p>.
- Se houver âncora compatível em links_internos, inclua 1 link interno natural no 2º parágrafo.
Concorrência (referência, NÃO copie):
{dados_concorrencia}
""".strip(),
        expected_output="HTML com 2–3 <p> e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}':
- 5–7 <h2>; até 2 <h3> por <h2> quando fizer sentido.
- Incluir 1 H2 "Erros comuns e armadilhas" e 1 H2 "Exemplos práticos / aplicação".
- Nomes específicos, sem redundância.
- Não incluir conteúdo; só <h2>/<h3>.
Baseie a cobertura na intenção de busca e no que os concorrentes ranqueados tratam:
{dados_concorrencia}
""".strip(),
        expected_output="Lista hierárquica com <h2>/<h3> apenas.",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos:
- Mín. 1200 palavras no post completo (será validado no unificador).
- <p> curtos (2–4 linhas); usar <ul><li> quando listar.
- Explique: o que é, por que importa, como fazer, exemplos reais.
- Variar semântica de '{palavra_chave}' sem stuffing.
- Sem autopromoção e sem CTA.
- Não inventar novos headings; usar apenas os fornecidos.
Concorrência (inspiração de tópicos, NÃO copiar):
{dados_concorrencia}
""".strip(),
        expected_output="HTML com <h2>/<h3>, <p> e <ul><li> preenchidos.",
        agent=agente_desenvolvimento
    )

    tarefa_conclusao = Task(
        description="""
Escreva a CONCLUSÃO:
- 1–2 <p> resumindo aprendizados e próximos passos práticos.
- Zero CTA.
- Inclua 1 link interno natural se ainda não houver link na conclusão.
""".strip(),
        expected_output="Conclusão em <p>, possivelmente com 1 link interno.",
        agent=agente_conclusao
    )

    tarefa_unificar = Task(
        description="""
Una introdução, corpo e conclusão em um único HTML (conteúdo do body, sem <body>).
Regras:
- Garantir coerência e zero repetição.
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body).",
        agent=agente_unificador
    )

    tarefa_linkagem = Task(
        description="""
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão):
- links_internos: [{titulo, url, anchor_sugerida}]
- links_externos (opcional): [{titulo, url, anchor_sugerida}]
Regras:
- >=3 links internos distribuídos: 1 na intro, 1–2 no corpo, 1 na conclusão (se aplicável).
- >=1 link externo autoritativo (se fornecido), no corpo, com target="_blank" rel="noopener noreferrer".
- Âncoras naturais; não repetir a mesma URL com a mesma âncora mais de 1x.
- Não linkar em headings; apenas <p> e <li>.
- Não quebrar HTML semântico; sem inline style.
Saída: HTML com linkagem aplicada.
""".strip(),
        expected_output="HTML com links internos/externos aplicados.",
        agent=agente_linkagem,
        context=[{"links_internos": links_internos, "links_externos": links_externos or []}]
    )

    tarefa_contato = Task(
        description="""
Anexar ao FINAL do HTML a assinatura institucional (sem alterar o conteúdo anterior):
<p>Clique aqui e agende uma reunião com nossos especialistas.<br>
<a href="https://api.whatsapp.com/send?phone=5511947974924&text=Oi!%20Encontrei%20seu%20site%20no%20Google%20e%20gostaria%20de%20mais%20informações." target="_blank" rel="noopener noreferrer">Fale conosco pelo WhatsApp</a></p>
<p><strong>Invictus Marketing</strong><br>Av. Casa Verde, 751 – São Paulo - SP</p>
""".strip(),
        expected_output="HTML final com assinatura adicionada.",
        agent=agente_contato
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT‑BR; clareza; tom Invictus (profissional, direto, útil).
- Frases longas; redundâncias; jargões; exemplos fracos.
- Coerência e distribuição de links; âncoras naturais; ausência de overstuffing de '{palavra_chave}'.
Saída: lista de melhorias acionáveis em bullets JSON‑like:
- {{"campo":"trecho/resumo","problema":"...","acao":"..."}}
""".strip(),
        expected_output="Bullets com melhorias acionáveis.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="""
Aplique TODAS as melhorias propostas, preservando:
- Estrutura semântica (<h2>/<h3>/<p>/<ul><li>/<a>).
- Linkagem já aplicada (ajuste âncora só se necessário).
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only).",
        agent=agente_executor
    )

    # ==== Crew ====
    crew_invictus = Crew(
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
    return crew_invictus
