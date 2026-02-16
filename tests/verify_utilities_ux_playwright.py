import asyncio
from playwright.async_api import async_playwright
import os
import time

async def run_test():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Increase timeout for streamlit to load
        page.set_default_timeout(60000)

        print("Connecting to dashboard...")
        try:
            await page.goto("http://localhost:8501")

            # Navigate to Utilities page
            # Streamlit sidebar links can be tricky. Let's try to click the Utilities link.
            print("Navigating to Utilities page...")
            # st.page_link uses anchors or buttons. Let's look for "Utilities"
            # In the dashboard, it's: st.page_link("pages/5_Utilities.py", label="Utilities", icon="🔧", ...)
            await page.get_by_text("Utilities").first.click()

            # Wait for the page to load
            await page.wait_for_selector("text=Utilities", timeout=30000)
            print("Utilities page loaded.")

            # Check for "Collect Logs" button and its checkbox
            # st.checkbox label: "I confirm I want to collect logs"
            collect_checkbox = page.get_by_text("I confirm I want to collect logs")
            await collect_checkbox.wait_for(state="visible")
            print("Found 'Collect Logs' checkbox.")

            # Check if button is disabled initially
            # st.button matches by text
            collect_button = page.get_by_role("button", name="🚀 Collect Logs")
            is_disabled = await collect_button.is_disabled()
            print(f"'Collect Logs' button disabled initially: {is_disabled}")

            # Click checkbox
            await collect_checkbox.click()
            time.sleep(1) # Wait for streamlit rerun

            is_disabled_after = await collect_button.is_disabled()
            print(f"'Collect Logs' button disabled after checking: {is_disabled_after}")

            # Take screenshot
            os.makedirs("screenshots", exist_ok=True)
            await page.screenshot(path="screenshots/utilities_interlock.png", full_page=True)
            print("Screenshot saved to screenshots/utilities_interlock.png")

            # Verify reconciliation interlock
            recon_checkbox = page.get_by_text("I confirm I want to unlock reconciliation tools")
            await recon_checkbox.wait_for(state="visible")
            print("Found reconciliation checkbox.")

            recon_button = page.get_by_role("button", name="🔄 Reconcile Council History")
            is_recon_disabled = await recon_button.is_disabled()
            print(f"Reconciliation button disabled initially: {is_recon_disabled}")

            await recon_checkbox.click()
            time.sleep(1)

            is_recon_disabled_after = await recon_button.is_disabled()
            print(f"Reconciliation button disabled after checking: {is_recon_disabled_after}")

            if is_disabled and is_disabled_after == False and is_recon_disabled and is_recon_disabled_after == False:
                print("UX Verification SUCCESS")
            else:
                print("UX Verification FAILURE")

        except Exception as e:
            print(f"Error during playwright test: {e}")
            # Take error screenshot
            await page.screenshot(path="screenshots/error.png")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run_test())
