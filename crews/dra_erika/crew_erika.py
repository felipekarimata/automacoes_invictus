import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
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


def build_crew_erika(tema: str, palavra_chave: str):
    dados_concorrencia = buscar_concorrentes_serpapi(palavra_chave)

    agente_intro = Agent(
        role="Redatora Dermatológica Integrativa",
        goal="Criar uma introdução técnica, acolhedora e conectada ao bem-estar do paciente",
        backstory="Especialista em comunicar temas de dermatologia clínica, cirúrgica e estética com ênfase em saúde e prevenção.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_h2 = Agent(
        role="Criadora de Subtítulos Dermatológicos",
        goal="Elaborar subtítulos <h2> para um artigo sobre dermatologia clínica, estética e cirúrgica, com clareza técnica",
        backstory="Especialista em estruturar conteúdos médicos para facilitar a leitura e reforçar autoridade profissional.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_meio_lista = Agent(
        role="Desenvolvedora de Conteúdo Dermatológico",
        goal="Desenvolver listas e parágrafos sobre procedimentos e cuidados dermatológicos, sempre com base nos subtítulos",
        backstory="Profissional com experiência em comunicar tecnologias, cuidados estéticos e dermatológicos com linguagem técnica e acessível.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_conclusao = Agent(
        role="Finalizadora de Conteúdos Dermatológicos",
        goal="Encerrar o artigo reforçando o cuidado dermatológico e a importância da avaliação profissional, sem CTA direta",
        backstory="Especialista em conclusões institucionais na área médica, focada em reforçar confiança e credibilidade.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_contato = Agent(
        role="Geradora de Assinatura Personalizada da Dra. Erika Voltan",
        goal="Adicionar uma assinatura final personalizada conforme o tema do artigo, incluindo links corretos de WhatsApp e Instagram",
        backstory="Responsável por reforçar a presença institucional da Dra. Erika, com foco em comunicação acolhedora e profissional.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_unificador = Agent(
        role="Unificadora de HTML Dermatológico",
        goal="Unificar todas as partes em HTML limpo, bem estruturado e pronto para publicação em WordPress",
        backstory="Especialista em estruturação de conteúdo médico para web, focando em legibilidade e organização.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_revisor = Agent(
        role="Revisora Técnica da Dra. Erika Voltan",
        goal="Revisar o conteúdo com atenção à clareza técnica, linguagem acolhedora e consistência com a abordagem integrativa",
        backstory="Revisora experiente em conteúdos médicos, estética e saúde, com foco na combinação de técnica e acolhimento.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_executor = Agent(
        role="Executor Técnico de Revisões",
        goal="Aplicar as sugestões de revisão mantendo o tom técnico, a estrutura e o formato HTML",
        backstory="Responsável por ajustar e finalizar conteúdos para publicação, garantindo fidelidade à proposta editorial.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_seo = Agent(
        role="Especialista em SEO Dermatológico",
        goal="Ajustar o conteúdo para SEO voltado a dermatologia integrativa e gerar meta description eficaz",
        backstory="Consultor de SEO para clínicas médicas, focado em otimizar conteúdo para buscas locais e médicas.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    agente_finalizador = Agent(
        role="Finalizador de Conteúdo para API",
        goal="Gerar JSON final com título, meta description e corpo em HTML formatado",
        backstory="Responsável por transformar conteúdos finais em JSON estruturado para publicação automatizada.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    tarefas = [
        Task(
            description=f"""Escreva uma introdução acolhedora para o tema '{tema}' com a palavra-chave '{palavra_chave}', considerando o foco em dermatologia clínica, estética e cirúrgica.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="2 parágrafos em HTML com linguagem técnica e acolhedora.",
            agent=agente_intro
        ),

        Task(
            description=f"""Crie subtítulos <h2> para um artigo sobre '{tema}', considerando as especialidades da Dra. Erika e este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Lista de subtítulos <h2> relevantes e claros.",
            agent=agente_meio_h2
        ),

        Task(
            description=f"""Desenvolva parágrafos <p> e listas <ul><li> com base nos subtítulos sobre '{tema}', explicando tratamentos, tecnologias e cuidados recomendados.
Considere este resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Conteúdo em HTML detalhado, claro e técnico.",
            agent=agente_meio_lista
        ),

        Task(
            description=f"""Finalize o artigo reforçando a importância do acompanhamento dermatológico e a abordagem integrativa, sem CTA direto.
Baseie-se neste resumo da concorrência:\n\n{dados_concorrencia}""",
            expected_output="Conclusão técnica e acolhedora, sem chamada para ação direta.",
            agent=agente_conclusao
        ),

        Task(
            description="""Inclua ao final do HTML a seguinte assinatura:

        <p><strong>Clique aqui e agende sua consulta para tratar flacidez corporal com a Dra. Érika Voltan!</strong><br>
        <a href="https://api.whatsapp.com/send?phone=5511966189853&text=Oi!%20Encontrei%20seu%20perfil%20no%20Google%20e%20gostaria%20de%20mais%20informações" target="_blank">https://api.whatsapp.com/send?phone=5511966189853&text=Oi!%20Encontrei%20seu%20perfil%20no%20Google%20e%20gostaria%20de%20mais%20informações</a></p>

        <p><a href="https://www.instagram.com/dra_erika_voltan/" target="_blank">Siga a Dra. Erika Voltan no Instagram</a></p>

        <p><strong>Dra. Érika Voltan — Dermatologista em Moema, São Paulo</strong></p>""",
            expected_output="HTML final com assinatura personalizada da Dra. Erika Voltan, incluindo link do WhatsApp e Instagram.",
            agent=agente_contato
        ),

        Task(
            description="Unifique todo o conteúdo em HTML limpo e organizado para publicação no WordPress.",
            expected_output="HTML único e fluido.",
            agent=agente_unificador
        ),

        Task(
            description="Revisar o conteúdo com foco em clareza técnica, tom acolhedor e consistência com a comunicação da Dra. Erika.",
            expected_output="Lista de ajustes e sugestões.",
            agent=agente_revisor
        ),

        Task(
            description="Aplicar as revisões mantendo a estrutura HTML e o tom do conteúdo.",
            expected_output="HTML revisado e finalizado.",
            agent=agente_executor
        ),

        Task(
            description="Otimizar o conteúdo final para SEO em dermatologia clínica, estética e cirúrgica. Gerar uma meta description com até 160 caracteres.",
            expected_output="HTML otimizado + meta description.",
            agent=agente_seo
        ),

        Task(
            description="Gerar JSON final com os campos: titulo, meta_description, html_body. Formatar para API.",
            expected_output="JSON pronto para integração.",
            agent=agente_finalizador
        )
    ]

    crew = Crew(
        agents=[
            agente_intro, agente_meio_h2, agente_meio_lista, agente_conclusao,
            agente_contato, agente_unificador, agente_revisor, agente_executor,
            agente_seo, agente_finalizador
        ],
        tasks=tarefas,
        verbose=True
    )

    return crew
