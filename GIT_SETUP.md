# Git Repository Setup

The local git repository has been initialized in the `cheatsheets/` directory.

## Connect to GitHub

To enable automatic pushing to GitHub, follow these steps:

### 1. Create a GitHub Repository

```bash
# Option A: Using GitHub CLI (if installed)
gh repo create ai-ml-cheatsheets --public --source=. --remote=origin

# Option B: Manual setup on GitHub.com
# 1. Go to https://github.com/new
# 2. Create a new repository named "ai-ml-cheatsheets"
# 3. Do NOT initialize with README (we already have one)
```

### 2. Add Remote and Push

```bash
cd cheatsheets

# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ai-ml-cheatsheets.git

# Rename branch to main (optional, if you prefer main over master)
git branch -M main

# Push initial commit
git push -u origin main
```

### 3. Verify Setup

```bash
# Check remote is configured
git remote -v

# Test push access
git push
```

## SSH Setup (Recommended)

For passwordless pushes, use SSH:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Add this key to GitHub: Settings → SSH and GPG keys → New SSH key

# Update remote to use SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ai-ml-cheatsheets.git
```

## Status

- ✅ Local git repository initialized
- ✅ Initial commit created
- ⏳ Remote repository connection (complete steps above)

Once remote is configured, the AI-ML Cheatsheet Creator agent will automatically:
1. Stage new cheatsheet files
2. Commit with descriptive messages
3. Push to the remote repository
