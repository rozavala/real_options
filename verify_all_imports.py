import sys
import os

# Mock streamlit
import types
sys.modules['streamlit'] = types.ModuleType('streamlit')
sys.modules['streamlit'].set_page_config = lambda **kwargs: None
sys.modules['streamlit'].title = lambda *args: None
sys.modules['streamlit'].markdown = lambda *args: None
sys.modules['streamlit'].caption = lambda *args: None
sys.modules['streamlit'].header = lambda *args: None
sys.modules['streamlit'].subheader = lambda *args: None
sys.modules['streamlit'].columns = lambda *args: [types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *args: None, metric=lambda *args, **kwargs: None) for _ in range(args[0])]
sys.modules['streamlit'].expander = lambda *args, **kwargs: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *args: None)
sys.modules['streamlit'].write = lambda *args: None
sys.modules['streamlit'].error = lambda *args: None
sys.modules['streamlit'].warning = lambda *args: None
sys.modules['streamlit'].info = lambda *args: None
sys.modules['streamlit'].success = lambda *args: None
sys.modules['streamlit'].stop = lambda *args: None
sys.modules['streamlit'].cache_data = lambda func=None, **kwargs: (lambda f: f) if func is None else func
sys.modules['streamlit'].sidebar = types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *args: None)
sys.modules['streamlit'].button = lambda *args, **kwargs: False
sys.modules['streamlit'].spinner = lambda *args: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *args: None)
sys.modules['streamlit'].dataframe = lambda *args, **kwargs: None
sys.modules['streamlit'].plotly_chart = lambda *args, **kwargs: None
sys.modules['streamlit'].selectbox = lambda *args, **kwargs: None

# Add root to path
sys.path.append(os.getcwd())

print("Importing dashboard_utils...")
try:
    import dashboard_utils
    print("SUCCESS: dashboard_utils")
except Exception as e:
    print(f"FAILURE: dashboard_utils: {e}")

print("Importing dashboard...")
try:
    import dashboard
    print("SUCCESS: dashboard")
except Exception as e:
    print(f"FAILURE: dashboard: {e}")

print("Importing pages/1_🦅_Cockpit.py...")
try:
    import pages.1_🦅_Cockpit
    print("SUCCESS: pages/1_🦅_Cockpit")
except ImportError:
    # Python doesn't like emojis in module names for import sometimes, or files starting with numbers
    # We will try to read and compile it manually
    with open('pages/1_🦅_Cockpit.py', 'r') as f:
        compile(f.read(), 'pages/1_🦅_Cockpit.py', 'exec')
    print("SUCCESS: pages/1_🦅_Cockpit (Syntax Check)")
except Exception as e:
    print(f"FAILURE: pages/1_🦅_Cockpit: {e}")

print("Importing pages/2_⚖️_The_Scorecard.py...")
try:
    with open('pages/2_⚖️_The_Scorecard.py', 'r') as f:
        compile(f.read(), 'pages/2_⚖️_The_Scorecard.py', 'exec')
    print("SUCCESS: pages/2_⚖️_The_Scorecard (Syntax Check)")
except Exception as e:
    print(f"FAILURE: pages/2_⚖️_The_Scorecard: {e}")

print("Importing pages/3_🧠_The_Council.py...")
try:
    with open('pages/3_🧠_The_Council.py', 'r') as f:
        compile(f.read(), 'pages/3_🧠_The_Council.py', 'exec')
    print("SUCCESS: pages/3_🧠_The_Council (Syntax Check)")
except Exception as e:
    print(f"FAILURE: pages/3_🧠_The_Council: {e}")

print("Importing pages/4_📈_Financials.py...")
try:
    with open('pages/4_📈_Financials.py', 'r') as f:
        compile(f.read(), 'pages/4_📈_Financials.py', 'exec')
    print("SUCCESS: pages/4_📈_Financials (Syntax Check)")
except Exception as e:
    print(f"FAILURE: pages/4_📈_Financials: {e}")
