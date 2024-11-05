import instaloader
import os

def download_instagram_posts(username, download_path):
    # Initialize instaloader
    loader = instaloader.Instaloader()

    # Set the download directory
    loader.dirname_pattern = os.path.join(download_path, '{profile}')
    
    # Download posts from the given username
    try:
        profile = instaloader.Profile.from_username(loader.context, username)

        for post in profile.get_posts():
            # Download post media (image/video)
            loader.download_post(post, target=profile.username)
            
            # Save the caption to a text file in the download path
            caption_filename = os.path.join(download_path, profile.username, f"{post.shortcode}_caption.txt")
            with open(caption_filename, "w", encoding="utf-8") as f:
                f.write(post.caption or "No caption")
                
        print(f"Downloaded posts and captions for {username} in {download_path}")
        
    except instaloader.exceptions.ProfileNotExistsException:
        print(f"The profile {username} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # List of Instagram usernames
    instagram_usernames = ['anxiety_depression___', '3.01am', 'quotesanxiety', 'artofpoets']
    download_location = './instagram_downloads'  # Replace this with your desired download path

    # Ensure the download path exists
    if not os.path.exists(download_location):
        os.makedirs(download_location)

    # Loop through each username and download posts
    for username in instagram_usernames:
        print(f"Starting download for {username}...")
        download_instagram_posts(username, download_location)
        print(f"Finished downloading posts for {username}.\n")

if __name__ == "__main__":
    main()
