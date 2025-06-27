#!/usr/bin/env python3
"""
Simple test to figure out correct TextClip parameters
"""

def test_textclip_params():
    try:
        from moviepy import TextClip
        
        # Test different parameter combinations to find what works
        test_cases = [
            {"text": "Test", "fontsize": 40, "color": "white"},
            {"text": "Test", "font_size": 40, "color": "white"}, 
            {"text": "Test", "size": 40, "color": "white"},
            {"text": "Test", "color": "white"},
            {"text": "Test", "method": "caption", "color": "white"},
        ]
        
        for i, params in enumerate(test_cases):
            try:
                print(f"Test {i+1}: {params}")
                clip = TextClip(**params)
                print(f"✅ Test {i+1} SUCCESS")
                
                # If this works, show available attributes
                print(f"Available methods: {[attr for attr in dir(clip) if not attr.startswith('_')][:10]}")
                break
                
            except Exception as e:
                print(f"❌ Test {i+1} FAILED: {e}")
                
    except Exception as e:
        print(f"Failed to import TextClip: {e}")

if __name__ == "__main__":
    test_textclip_params() 