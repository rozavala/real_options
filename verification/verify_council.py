
from playwright.sync_api import Page, expect, sync_playwright
import time

def test_council_page(page: Page):
    """Verify The Council page renders safely without unsafe_allow_html."""

    # 1. Navigate to the app
    page.goto("http://localhost:8501")

    # 2. Wait for loading
    page.wait_for_timeout(5000)

    # 3. Navigate to "The Council" page
    # Streamlit sidebar navigation
    page.get_by_text("The Council").click()
    page.wait_for_timeout(3000)

    # 4. Scroll to Forensic Context section
    # Look for "Forensic Context" header
    expect(page.get_by_text("Forensic Context")).to_be_visible()

    # 5. Take Screenshot of Forensic Section (Trigger Type, Catalyst, etc)
    page.screenshot(path="/home/jules/verification/council_forensic.png")

    # 6. Scroll to Master Decision section
    expect(page.get_by_text("Master Decision")).to_be_visible()

    # 7. Take Screenshot of Master Decision
    page.screenshot(path="/home/jules/verification/council_master_decision.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            test_council_page(page)
        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="/home/jules/verification/error.png")
        finally:
            browser.close()
