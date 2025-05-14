"""
Streamlit UI for Travel Planning Agent

This script creates a chat-like interface using Streamlit to interact with the travel planning agent.
The user can enter a destination and receive a travel plan in a conversational format with
streaming/typewriter effects for a more engaging experience.
"""
import streamlit as st
import traceback
import time
import random
import json
from travel_agent import plan_travel, run_attractions_agent, run_route_planner_agent, run_finalizer_agent
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Agente de Viagem IA", page_icon="✈️", layout="centered")
st.title("✈️ Agente de Viagem com IA")
st.markdown("""
Converse com o agente para planejar sua próxima viagem!\
Digite o destino desejado e receba um roteiro completo com atualizações em tempo real.
""")

# Configurações de exibição e velocidade
with st.sidebar:
    st.header("⚙️ Configurações")
    
    speed_options = {
        "Rápida (sem efeitos)": None,
        "Normal": 0.01, 
        "Lenta": 0.03
    }
    
    if "typing_speed" not in st.session_state:
        st.session_state["typing_speed"] = 0.01  # Default: Normal
    
    selected_speed = st.radio(
        "Velocidade de exibição:",
        list(speed_options.keys()),
        index=1  # Default: Normal
    )
    
    st.session_state["typing_speed"] = speed_options[selected_speed]
    
    st.divider()
    st.caption("Desenvolvido com Streamlit, LangChain e Azure OpenAI")

def typewriter_effect(container, text, speed=0.02, complete_instantly=False):
    """
    Create a typewriter effect for the text in the given container.
    
    Args:
        container: The Streamlit container to write to
        text: The text to display
        speed: The speed of typing (lower is faster)
        complete_instantly: If True, will display the full text without animation
    """
    if complete_instantly:
        container.markdown(text)
        return
    
    placeholder = container.empty()
    
    # Simulate typing with realistic variable speeds
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(full_text + "▌")  # Add cursor
        # Variable speed: faster for normal chars, slower for punctuation
        delay = speed
        if char in ".!?":
            delay *= 3
        elif char in ",;:":
            delay *= 2
        elif char == "\n":
            delay *= 4
        time.sleep(delay)
    
    # Final text without cursor
    placeholder.markdown(full_text)
    return full_text

def loading_animation(container, message, duration=3):
    """Show a loading animation with dots in the container"""
    start_time = time.time()
    while time.time() - start_time < duration:
        for i in range(4):
            dots = "." * i
            container.markdown(f"{message}{dots}")
            time.sleep(0.3)

def format_attractions_list(attractions):
    """Format attractions as a nice list with emojis"""
    icons = ["🏛️", "🗿", "🏰", "🌉", "🏯", "🏙️", "⛪", "🌆", "🏝️", "🗼"]
    result = ""
    
    for i, attr in enumerate(attractions):
        icon = icons[i % len(icons)]
        name = attr.get("name", "Atração")
        desc = attr.get("description", "")
        time_needed = attr.get("recommended_time", "")
        
        result += f"{icon} **{name}** "
        if time_needed:
            result += f"({time_needed})\n"
        else:
            result += "\n"
            
        if desc:
            result += f"   {desc}\n\n"
    
    return result

def format_itinerary_preview(route_plan):
    """Create a preview of the itinerary days"""
    result = ""
    
    if not route_plan or "day_by_day_itinerary" not in route_plan:
        return "📅 Itinerário sendo elaborado..."
    
    days = route_plan.get("day_by_day_itinerary", [])
    day_count = len(days)
    
    result += f"📅 **Itinerário de {day_count} dias criado!**\n\n"
    
    # Just show first day as preview
    if days:
        first_day = days[0]
        result += f"**Dia {first_day.get('day', 1)} (preview):**\n"
        activities = first_day.get("activities", [])
        
        for act in activities:
            if isinstance(act, dict) and "activity" in act:
                result += f"- {act.get('time', '')}: {act.get('activity', '')}\n"
            else:
                result += f"- {act}\n"
    
    result += "\n... mais detalhes no plano final\n"
    return result

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat history UI
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Input box for user
if prompt := st.chat_input("Digite o destino da sua viagem ou uma pergunta..."):
    # Adiciona mensagem do usuário
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Prepara o chat do assistente
    assistant_msg = st.chat_message("assistant")
    message_container = assistant_msg.container()
    
    try:
        # Inicializa o estado
        state = {
            "messages": [HumanMessage(content=f"I want to plan a trip to {prompt}")],
            "destination": prompt,
            "attractions": [],
            "route_plan": {},
            "final_plan": {}
        }
        
        # Barra de progresso para acompanhar as etapas
        progress_text = "Planejando sua viagem"
        progress_bar = st.progress(0, text=progress_text)
        
        # Determina se deve mostrar efeitos de digitação ou não
        typing_speed = st.session_state.get("typing_speed", 0.01)
        complete_instantly = typing_speed is None
        
        # Etapa 1: Encontrando atrações com efeito de digitação (25% do processo)
        typewriter_effect(
            message_container, 
            f"🔎 Buscando atrações turísticas para **{prompt}**...",
            speed=typing_speed,
            complete_instantly=complete_instantly
        )
        
        progress_bar.progress(10, text="Pesquisando atrações turísticas...")
        
        # Simula processamento com animação apenas se não for instantâneo
        loading_placeholder = message_container.empty()
        if not complete_instantly:
            loading_animation(loading_placeholder, "🔄 Pesquisando destinos")
        
        # Executando o agente
        state = run_attractions_agent(state)
        loading_placeholder.empty()
        progress_bar.progress(25, text="Atrações encontradas!")
        
        # Atualiza o texto com as atrações encontradas
        attractions_text = f"🔎 Buscando atrações turísticas para **{prompt}**...\n\n"
        attractions_text += f"✅ **{len(state['attractions'])} atrações encontradas!**\n\n"
        attractions_text += format_attractions_list(state['attractions'][:5])  # Limita a 5 para não ficar muito grande
        
        typewriter_effect(
            message_container, 
            attractions_text,
            speed=typing_speed,
            complete_instantly=complete_instantly
        )
        
        # Etapa 2: Planejando rota com efeito de digitação (60% do processo)
        attractions_text += "\n\n🗺️ **Planejando o roteiro entre as atrações...**"
        typewriter_effect(
            message_container, 
            attractions_text,
            speed=typing_speed,
            complete_instantly=complete_instantly
        )
        
        progress_bar.progress(35, text="Calculando melhores rotas...")
        
        # Simula processamento com animação apenas se não for instantâneo
        loading_placeholder = message_container.empty()
        if not complete_instantly:
            loading_animation(loading_placeholder, "🔄 Calculando rotas")
        
        # Executando o agente
        state = run_route_planner_agent(state)
        loading_placeholder.empty()
        progress_bar.progress(60, text="Rotas planejadas!")
        
        # Atualiza o texto com o plano de rota 
        route_text = attractions_text + "\n\n"
        route_text += f"✅ **Rota planejada**\n\n"
        route_text += format_itinerary_preview(state['route_plan'])
        
        typewriter_effect(
            message_container, 
            route_text,
            speed=typing_speed,
            complete_instantly=complete_instantly
        )
        
        # Etapa 3: Finalizando plano com efeito de digitação (100% do processo)
        route_text += "\n\n📋 **Finalizando o plano de viagem...**"
        typewriter_effect(
            message_container, 
            route_text,
            speed=typing_speed,
            complete_instantly=complete_instantly
        )
        
        progress_bar.progress(70, text="Compilando plano final...")
        
        # Simula processamento com animação
        loading_placeholder = message_container.empty()
        if not complete_instantly:
            loading_animation(loading_placeholder, "🔄 Compilando plano final")
        
        # Executando o agente  
        state = run_finalizer_agent(state)
        loading_placeholder.empty()
        progress_bar.progress(90, text="Plano finalizado!")
        
        # Prepara a resposta final com efeito de digitação
        if state.get("final_plan"):
            plan = state["final_plan"]
            resposta = route_text + "\n\n"
            resposta += "✅ **Plano finalizado!**\n\n"
            resposta += "---\n\n"
            resposta += f"**🌍 Destino:** {state['destination']}\n\n"
            
            if "destination_overview" in plan:
                resposta += f"{plan['destination_overview']}\n\n"
                
            if "itinerary" in plan:
                resposta += "**📅 Itinerário:**\n"
                for day in plan["itinerary"]:
                    resposta += f"- **Dia {day.get('day', '?')}:**\n"
                    for act in day.get("activities", []):
                        resposta += f"    - {act}\n"
            
            if "practical_tips" in plan:
                resposta += f"\n**💡 Dicas úteis:** {plan['practical_tips']}"
                
            if "cost_estimate" in plan:
                resposta += f"\n\n**💰 Custo estimado:** {plan['cost_estimate']}"
                
            if "packing_suggestions" in plan:
                resposta += f"\n\n**🧳 Sugestões de bagagem:** {plan['packing_suggestions']}"
            
            # Atualiza container com a resposta final usando efeito de digitação
            typing_speed_final = typing_speed if typing_speed is None else typing_speed * 0.7  # Um pouco mais rápido para o texto final
            typewriter_effect(
                message_container, 
                resposta,
                speed=typing_speed_final,
                complete_instantly=complete_instantly
            )
            
            # Completa a barra de progresso
            progress_bar.progress(100, text="Plano de viagem completo!")
                
            # Salva a mensagem final no histórico
            st.session_state["messages"].append({"role": "assistant", "content": resposta})
        else:
            msg = route_text + "\n\n❌ **Não foi possível gerar um plano de viagem completo.**"
            typewriter_effect(
                message_container, 
                msg,
                speed=typing_speed,
                complete_instantly=complete_instantly
            )
            progress_bar.progress(100, text="Processo finalizado com erros.")
            st.session_state["messages"].append({"role": "assistant", "content": msg})
    
    except Exception as e:
        tb = traceback.format_exc()
        erro = f"❌ **Erro inesperado ao planejar viagem para {prompt}**\n\n"
        erro += "Ocorreu um problema ao gerar seu plano de viagem. Isso pode acontecer por várias razões:\n\n"
        erro += "- Problema temporário de conexão com o serviço Azure OpenAI\n"
        erro += "- Destino muito específico ou com poucos dados disponíveis\n"
        erro += "- Problemas internos na geração do plano\n\n"
        erro += f"Detalhes técnicos: `{str(e)[:100]}...`"
        
        # Se houver progresso definido, atualize-o
        if 'progress_bar' in locals():
            progress_bar.progress(100, text="Erro ao processar solicitação")
            
        # Use o typewriter com a velocidade selecionada pelo usuário
        typing_speed = st.session_state.get("typing_speed", 0.01)
        complete_instantly = typing_speed is None
        
        typewriter_effect(
            message_container, 
            erro,
            speed=typing_speed,
            complete_instantly=complete_instantly
        )
        
        # Log completo para debug
        st.session_state["messages"].append({"role": "assistant", "content": erro})
        print(f"Erro detalhado: {e}\n{tb}")