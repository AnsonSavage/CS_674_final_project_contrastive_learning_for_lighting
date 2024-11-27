import json
import requests
from bs4 import BeautifulSoup

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

def save_to_json(data, filename='asset_metadata.json'):
    """
    Save extracted data to a JSON file.
    
    Args:
        data (dict): Dictionary of categories and tags
        filename (str): Name of the output JSON file
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def scrape_polyhaven_asset(url):
    """
    Main function to scrape a Poly Haven asset page.
    
    Args:
        url (str): URL of the Poly Haven asset page
    
    Returns:
        dict: Extracted asset metadata
    """
    # Fetch HTML content
    html_content = fetch_html_content(url)

    if not html_content:
        return None
    
    # Extract tags and categories
    asset_metadata = extract_tags_and_categories(html_content)
    
    # Save to JSON file
    save_to_json(asset_metadata)
    
    # Print the results
    print("Asset Metadata:")
    print(json.dumps(asset_metadata, indent=2))
    
    return asset_metadata

def main():
    # Example usage
    url = 'https://polyhaven.com/a/rogland_clear_night'
    scrape_polyhaven_asset(url)

if __name__ == '__main__':
    main()

# Requirements:
# pip install beautifulsoup4 requests