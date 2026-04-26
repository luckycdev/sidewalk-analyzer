import requests

def get_overture_link_via_api(gers_id, theme="transportation", otype="segment"):
    url = f"https://geocoder.bradr.dev/id/{gers_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        bbox = data.get("bbox")
        if not bbox:
            return "Error: ID found but no bounding box returned."
        
        # Calculate the center of the bounding box
        lat = (bbox["ymin"] + bbox["ymax"]) / 2
        lon = (bbox["xmin"] + bbox["xmax"]) / 2
        
        # Format the final Explorer link
        zoom = 16.75
        link = f"https://explore.overturemaps.org/?mode=explore&feature={theme}.{otype}.{gers_id}#{zoom}/{lat:.6f}/{lon:.6f}"
        return link

    except requests.exceptions.RequestException as e:
        return f"API Lookup failed: {e}"

if __name__ == "__main__":
    target_id = "796f6f15-aef7-4b33-b733-5710d91497e1"
    print("\n--- Generated Link ---")
    print(get_overture_link_via_api(target_id))