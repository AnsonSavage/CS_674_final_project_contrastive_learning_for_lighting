import json
import requests
from bs4 import BeautifulSoup
import argparse
import os

def fetch_html_content(url):
    """
    Fetch HTML content from a given URL.
    
    Args:
        url (str): The URL to fetch
    
    Returns:
        str: HTML content of the page
    """
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

def extract_tags_and_categories(html_content):
    """
    Extract categories and tags from Poly Haven asset page HTML.

    Note, this depends on a unique <strong> HTML element containing the text "Categories" and "Tags".
    
    Args:
        html_content (str): The HTML content of the page
    
    Returns:
        dict: A dictionary containing categories and tags
    """
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    
    # Extract categories
    all_strong_sections = soup.find_all('strong')
    for strong_section in all_strong_sections:
        # If it contains "Categories" then we have found the section. We then need to get its parent div
        if "Categories" in strong_section.text:
            # category_span = strong_section.find_parent('span', class_='AssetPage_infoItem__rffWr')
            category_span = strong_section.parent
            break

    categories = []
    if category_span:
        category_tags = category_span.find_all('div', class_="AssetPage_tag__GHBX_")
        categories = [tag.text.strip() for tag in category_tags]


    
    # Extract tags
    for strong_section in all_strong_sections:
        # If it contains "Tags" then we have found the section. We then need to get its parent div
        if "Tags" in strong_section.text:
            # tags_span = strong_section.find_parent('span', class_='AssetPage_infoItem__rffWr')
            tags_span = strong_section.parent
            break
    tags = []
    if tags_span:
        tag_tags = tags_span.find_all('div', class_='AssetPage_tag__GHBX_')
        tags = [tag.text.strip() for tag in tag_tags]
    
    
    # Prepare output
    output = {
        "categories": categories,
        "tags": tags
    }
    
    return output

def save_to_json(data, directory, hdri_name):
    """
    Save extracted data to a JSON file inside the specified directory with HDRI name prefix.
    
    Args:
        data (dict): Dictionary of categories and tags
        directory (str): Directory to save the JSON file
        hdri_name (str): Name prefix for the JSON file
    """
    filepath = os.path.join(directory, f"{hdri_name}_asset_metadata.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def scrape_polyhaven_asset(url, directory):
    """
    Scrape a single Poly Haven asset page and save metadata.
    
    Args:
        url (str): URL of the Poly Haven asset page
        directory (str): Directory to save the JSON file
    """
    hdri_name = url.rstrip('/').split('/')[-1].lower()
    hdri_dir = os.path.join(directory, hdri_name)
    os.makedirs(hdri_dir, exist_ok=True)

    html_content = fetch_html_content(url)

    if not html_content:
        return
    
    asset_metadata = extract_tags_and_categories(html_content)
    
    save_to_json(asset_metadata, hdri_dir, hdri_name)
    
    print(f"Asset Metadata for '{hdri_name}':")
    print(json.dumps(asset_metadata, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Scrape HDRI metadata from Poly Haven.')
    parser.add_argument('directory', type=str, help='Directory to save HDRI metadata')
    parser.add_argument('urls', nargs='+', help='List of Poly Haven asset URLs to scrape')
    args = parser.parse_args()

    for url in args.urls:
        scrape_polyhaven_asset(url, args.directory)

if __name__ == '__main__':
    main()
