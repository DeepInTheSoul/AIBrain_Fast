import os
from streamlit_option_menu import option_menu
import streamlit as st
from chat import chat_page
from rag import rag_page


if __name__ == '__main__':
    st.set_page_config(
        "My Chat App",
    )
    pages = {
        "对话": {
            "icon": "chat",
            "func": chat_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": rag_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "1.jpg"
            ),
            use_column_width=True
        )

        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"]()
