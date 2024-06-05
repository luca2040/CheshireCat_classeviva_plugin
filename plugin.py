from cat.mad_hatter.decorators import hook, tool
from cat.plugins.CC_plugin_foglietti_illustrativi.functions import *


@hook
def agent_prompt_prefix(prefix, cat):

    prefix = """
    Sei un farmacista, e rispondi in modo professionale.
    Non rispondi con informazioni che non ti sono state fornite esplicitamente.
    Non rispondi a domande inappropriate.
    Ad ogni domanda rispondi nel modo più completo e preciso possibile.
    """

    return prefix


@hook
def before_cat_recalls_declarative_memories(declarative_recall_config, cat):
    medicine = "fi_abilify_1mg_sol.txt"
    # history = cat.working_memory.history
    # max_index = len(history) - 1
    # index_now = max_index

    # while medicine == -1:
    #     medicine = get_query_medicine(cat, names_from_metadata(cat), history[index_now])

    #     index_now -= 1
    #     if index_now < 0:
    #         medicine = ""

    declarative_recall_config["metadata"] = {"source": medicine}

    return declarative_recall_config


@tool(return_direct=True)
def how_many_medicines_known(tool_input, cat):
    """Reply only to the question "How many medicines you know?" or to others which are stricly similar.
    Ignore generic questions about medicines.
    Input is always None"""

    names_list = names_from_metadata(cat)

    out_text = ""
    for number, name in enumerate(names_list):
        out_text += f"{number+1}. {name}\n"

    return out_text
