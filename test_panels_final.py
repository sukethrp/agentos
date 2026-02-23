#!/usr/bin/env python3
"""
Final comprehensive panel test with proper element clicking
"""
import asyncio
import sys
from playwright.async_api import async_playwright

async def test_all_panels():
    """Test all sidebar panels systematically"""
    
    print("\n" + "="*70)
    print("AGENTOS COMPREHENSIVE PANEL TESTING")
    print("="*70 + "\n")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Track console messages
        console_errors = []
        
        def handle_console(msg):
            if msg.type == 'error':
                text = msg.text
                # Filter out common non-critical errors
                if 'Failed to load resource' not in text and 'favicon' not in text:
                    console_errors.append(text)
        
        page.on('console', handle_console)
        
        page_errors = []
        def handle_page_error(error):
            page_errors.append(str(error))
        page.on('pageerror', handle_page_error)
        
        results = []
        
        try:
            print("📍 Navigating to http://localhost:8000...")
            await page.goto('http://localhost:8000', wait_until='load')
            await page.wait_for_timeout(3000)  # Wait for JS to load
            print("✅ Page loaded\n")
            
            # Get all nav items
            nav_items_data = await page.evaluate('''() => {
                const items = Array.from(document.querySelectorAll('.nav-item'));
                return items.map((item, index) => ({
                    index,
                    text: item.innerText.trim(),
                    onclick: item.getAttribute('onclick')
                }));
            }''')
            
            print(f"Found {len(nav_items_data)} navigation items\n")
            
            for item_data in nav_items_data:
                index = item_data['index']
                text = item_data['text']
                onclick = item_data['onclick']
                
                # Extract panel ID from onclick
                panel_id = None
                if onclick:
                    import re
                    match = re.search(r"showPanel\('([^']+)'", onclick)
                    if match:
                        panel_id = match.group(1)
                
                if not panel_id:
                    print(f"{index+1}. {text}: SKIPPED (no panel ID)")
                    continue
                
                print(f"{index+1}. Testing: {text}")
                
                panel_errors = []
                console_before = len(console_errors)
                
                try:
                    # Click the nav item by index
                    await page.evaluate(f'''() => {{
                        const items = document.querySelectorAll('.nav-item');
                        items[{index}].click();
                    }}''')
                    
                    await page.wait_for_timeout(600)
                    
                    # Check if panel is visible
                    panel_check = await page.evaluate(f'''() => {{
                        const panel = document.getElementById('panel-{panel_id}');
                        if (!panel) return {{exists: false}};
                        const isActive = panel.classList.contains('active');
                        const display = window.getComputedStyle(panel).display;
                        return {{
                            exists: true,
                            isActive,
                            display,
                            hasContent: panel.children.length > 0
                        }};
                    }}''')
                    
                    if not panel_check['exists']:
                        panel_errors.append(f"Panel not found: panel-{panel_id}")
                    elif not panel_check['isActive']:
                        panel_errors.append("Panel not active")
                    elif panel_check['display'] == 'none':
                        panel_errors.append("Panel display is 'none'")
                    elif not panel_check['hasContent']:
                        panel_errors.append("Panel has no content")
                    
                    # Check for new console errors
                    new_errors = console_errors[console_before:]
                    if new_errors:
                        for err in new_errors[:2]:  # Show first 2
                            panel_errors.append(f"Console: {err[:80]}")
                    
                    # Try to interact with panel (click refresh if exists)
                    try:
                        clicked = await page.evaluate(f'''() => {{
                            const panel = document.getElementById('panel-{panel_id}');
                            if (!panel) return false;
                            const btn = panel.querySelector('button');
                            if (btn && btn.innerText.includes('Refresh')) {{
                                btn.click();
                                return true;
                            }}
                            return false;
                        }}''')
                        if clicked:
                            await page.wait_for_timeout(300)
                    except:
                        pass
                    
                    # Status
                    if panel_errors:
                        status = "ERROR"
                        print(f"   ❌ ERROR: {panel_errors[0]}")
                    else:
                        status = "OK"
                        print(f"   ✅ OK")
                    
                    results.append({
                        "name": text,
                        "status": status,
                        "errors": panel_errors
                    })
                
                except Exception as e:
                    panel_errors.append(f"Exception: {str(e)[:80]}")
                    print(f"   ❌ ERROR: {str(e)[:80]}")
                    results.append({
                        "name": text,
                        "status": "ERROR",
                        "errors": panel_errors
                    })
            
            # Summary
            print("\n" + "="*70)
            print("📊 STRUCTURED SUMMARY")
            print("="*70 + "\n")
            
            for result in results:
                if result["status"] == "OK":
                    print(f"- {result['name']}: OK")
                else:
                    error_desc = result["errors"][0] if result["errors"] else "Unknown error"
                    print(f"- {result['name']}: ERROR: {error_desc}")
            
            print("\n" + "="*70)
            print("STATISTICS")
            print("="*70)
            
            ok_count = sum(1 for r in results if r["status"] == "OK")
            error_count = sum(1 for r in results if r["status"] == "ERROR")
            
            print(f"\nTotal Panels Tested: {len(results)}")
            print(f"✅ OK: {ok_count} ({ok_count*100//len(results) if results else 0}%)")
            print(f"❌ ERROR: {error_count} ({error_count*100//len(results) if results else 0}%)")
            
            if console_errors:
                unique_errors = list(set(console_errors))
                print(f"\n⚠️  Console Errors: {len(unique_errors)} unique")
                for err in unique_errors[:3]:
                    print(f"   - {err[:100]}")
            else:
                print("\n✅ No JavaScript console errors detected")
            
            if page_errors:
                print(f"\n⚠️  Page Errors: {len(page_errors)}")
                for err in page_errors[:3]:
                    print(f"   - {err[:100]}")
            
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
