#!/usr/bin/env python3
"""
Working panel test that properly waits for page load
"""
import asyncio
from playwright.async_api import async_playwright

async def test_panels():
    print("\n" + "="*70)
    print("AGENTOS PANEL TESTING - SYSTEMATIC CHECK")
    print("="*70 + "\n")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        console_errors = []
        def log_console(msg):
            if msg.type == 'error' and 'favicon' not in msg.text:
                console_errors.append(msg.text)
        page.on('console', log_console)
        
        try:
            print("📍 Loading http://localhost:8000...")
            await page.goto('http://localhost:8000')
            
            # Wait for the showPanel function to be defined
            await page.wait_for_function('typeof showPanel === "function"', timeout=10000)
            print("✅ Page and JavaScript loaded\n")
            
            # Get all panels
            panels_info = await page.evaluate('''() => {
                const navItems = Array.from(document.querySelectorAll('.nav-item'));
                return navItems.map(item => {
                    const onclick = item.getAttribute('onclick');
                    const match = onclick ? onclick.match(/showPanel\\('([^']+)'/) : null;
                    return {
                        text: item.innerText.trim(),
                        panelId: match ? match[1] : null
                    };
                }).filter(item => item.panelId);
            }''')
            
            results = []
            
            for i, panel_info in enumerate(panels_info, 1):
                name = panel_info['text']
                panel_id = panel_info['panelId']
                
                print(f"{i}. Testing: {name}")
                
                errors = []
                
                try:
                    # Call showPanel
                    await page.evaluate(f"showPanel('{panel_id}')")
                    await page.wait_for_timeout(400)
                    
                    # Check panel state
                    state = await page.evaluate(f'''() => {{
                        const panel = document.getElementById('panel-{panel_id}');
                        if (!panel) return {{found: false}};
                        return {{
                            found: true,
                            active: panel.classList.contains('active'),
                            visible: panel.offsetParent !== null,
                            hasContent: panel.children.length > 0
                        }};
                    }}''')
                    
                    if not state['found']:
                        errors.append(f"Panel not found: panel-{panel_id}")
                    elif not state['active']:
                        errors.append("Panel not active")
                    elif not state['visible']:
                        errors.append("Panel not visible")
                    
                    status = "OK" if not errors else "ERROR"
                    
                    if status == "OK":
                        print(f"   ✅ OK")
                    else:
                        print(f"   ❌ ERROR: {errors[0]}")
                    
                    results.append({
                        "name": name,
                        "status": status,
                        "errors": errors
                    })
                
                except Exception as e:
                    print(f"   ❌ ERROR: {str(e)[:60]}")
                    results.append({
                        "name": name,
                        "status": "ERROR",
                        "errors": [str(e)]
                    })
            
            # Summary
            print("\n" + "="*70)
            print("📊 STRUCTURED SUMMARY")
            print("="*70 + "\n")
            
            for r in results:
                if r["status"] == "OK":
                    print(f"- {r['name']}: OK")
                else:
                    desc = r["errors"][0] if r["errors"] else "Unknown"
                    print(f"- {r['name']}: ERROR: {desc}")
            
            ok = sum(1 for r in results if r["status"] == "OK")
            total = len(results)
            
            print("\n" + "="*70)
            print(f"✅ OK: {ok}/{total} ({ok*100//total if total else 0}%)")
            print(f"❌ ERROR: {total-ok}/{total}")
            
            if console_errors:
                print(f"\n⚠️  Console Errors: {len(console_errors)}")
                for err in console_errors[:3]:
                    print(f"   - {err[:80]}")
            else:
                print("\n✅ No console errors")
            
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_panels())
