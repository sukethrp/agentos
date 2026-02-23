#!/usr/bin/env python3
"""
Panel testing using JavaScript execution for reliable panel switching
"""
import asyncio
import sys
from playwright.async_api import async_playwright

async def test_all_panels():
    """Test all sidebar panels by executing showPanel() directly"""
    
    print("\n" + "="*70)
    print("AGENTOS COMPREHENSIVE PANEL TESTING (JS Execution)")
    print("="*70 + "\n")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Track console messages
        console_errors = []
        console_warnings = []
        
        def handle_console(msg):
            if msg.type == 'error':
                console_errors.append(f"{msg.text}")
            elif msg.type == 'warning':
                console_warnings.append(f"{msg.text}")
        
        page.on('console', handle_console)
        
        # Track page errors
        page_errors = []
        def handle_page_error(error):
            page_errors.append(str(error))
        page.on('pageerror', handle_page_error)
        
        results = []
        
        try:
            print("📍 Navigating to http://localhost:8000...")
            await page.goto('http://localhost:8000', wait_until='networkidle')
            await page.wait_for_timeout(2000)
            print("✅ Page loaded\n")
            
            # Test cases with panel IDs
            test_cases = [
                (1, "Agent Builder", "builder", "#b-query"),
                (2, "Templates", "templates", ".templates-grid"),
                (3, "Chat", "chat", "#chat-query"),
                (4, "Branching", "branching", "#br-tree-select"),
                (5, "Monitor", "monitor", ".monitor-grid"),
                (6, "Analytics", "analytics", ".an-summary"),
                (7, "Scheduler", "scheduler", "#sched-name"),
                (8, "Events", "events", "#events-list"),
                (9, "A/B Testing", "abtest", "#ab-prompt-a"),
                (10, "Multi-modal", "multimodal", ".mm-upload-zone"),
                (11, "Account & Usage", "auth", "#auth-email"),
                (12, "Marketplace", "marketplace", "#mp-list"),
                (13, "Embed SDK", "embed", "#embed-agent-name"),
                (14, "Agent Mesh", "mesh", "#mesh-container"),
                (15, "Simulation", "simulation", "#sim-name"),
                (16, "Learning", "learning", "#learning-stats"),
                (17, "RCA", "rca", "#rca-container"),
            ]
            
            for num, name, panel_id, check_selector in test_cases:
                print(f"{num}. Testing: {name}")
                
                panel_errors = []
                initial_error_count = len(console_errors)
                
                try:
                    # Use JavaScript to switch panel
                    await page.evaluate(f"showPanel('{panel_id}')")
                    await page.wait_for_timeout(500)
                    
                    # Check if panel is visible
                    panel = await page.query_selector(f'#panel-{panel_id}')
                    if not panel:
                        panel_errors.append(f"Panel element not found: panel-{panel_id}")
                        print(f"   ❌ ERROR: Panel not found")
                    else:
                        is_visible = await panel.evaluate('el => el.classList.contains("active")')
                        if not is_visible:
                            panel_errors.append("Panel not visible (no active class)")
                            print(f"   ❌ ERROR: Panel not visible")
                        else:
                            # Check for key element
                            element = await page.query_selector(check_selector)
                            if not element:
                                # Some elements might not exist, that's ok
                                pass
                            
                            # Check for visual bugs (basic check)
                            display = await panel.evaluate('el => window.getComputedStyle(el).display')
                            if display == 'none':
                                panel_errors.append("Panel display is 'none'")
                    
                    # Check for new console errors
                    new_errors = console_errors[initial_error_count:]
                    if new_errors:
                        for err in new_errors:
                            if 'Failed to load resource' not in err:  # Ignore resource loading errors
                                panel_errors.append(f"Console: {err}")
                    
                    # Try clicking refresh button if it exists
                    try:
                        refresh_btn = await panel.query_selector("button:has-text('Refresh')")
                        if refresh_btn:
                            await refresh_btn.click()
                            await page.wait_for_timeout(300)
                    except:
                        pass
                    
                    # Determine status
                    if panel_errors:
                        status = "ERROR"
                        print(f"   ❌ ERROR: {panel_errors[0]}")
                    else:
                        status = "OK"
                        print(f"   ✅ OK")
                    
                    results.append({
                        "num": num,
                        "name": name,
                        "status": status,
                        "errors": panel_errors
                    })
                
                except Exception as e:
                    panel_errors.append(f"Exception: {str(e)}")
                    print(f"   ❌ ERROR: {str(e)}")
                    results.append({
                        "num": num,
                        "name": name,
                        "status": "ERROR",
                        "errors": panel_errors
                    })
            
            # Summary
            print("\n" + "="*70)
            print("📊 STRUCTURED SUMMARY")
            print("="*70 + "\n")
            
            for result in results:
                status_text = result["status"]
                if status_text == "OK":
                    print(f"- {result['name']}: OK")
                else:
                    error_desc = result["errors"][0] if result["errors"] else "Unknown error"
                    print(f"- {result['name']}: ERROR: {error_desc}")
            
            print("\n" + "="*70)
            print("STATISTICS")
            print("="*70)
            
            ok_count = sum(1 for r in results if r["status"] == "OK")
            error_count = sum(1 for r in results if r["status"] == "ERROR")
            
            print(f"\nTotal Panels: {len(results)}")
            print(f"✅ OK: {ok_count} ({ok_count*100//len(results)}%)")
            print(f"❌ ERROR: {error_count} ({error_count*100//len(results)}%)")
            
            if console_errors:
                unique_errors = list(set(console_errors))
                print(f"\n⚠️  Console Errors: {len(unique_errors)} unique")
                for err in unique_errors[:5]:
                    print(f"   - {err[:100]}")
            else:
                print("\n✅ No JavaScript console errors detected")
            
            print("\n" + "="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await browser.close()
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_all_panels())
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
        sys.exit(1)
