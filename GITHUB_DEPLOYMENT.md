# FreshTrack Backend - GitHub Deployment Guide

## 🚀 **Complete Step-by-Step Guide**

### **Prerequisites**
- GitHub account
- Git installed on your computer
- Backend folder ready

---

## 📋 **Step 1: Create GitHub Repository**

1. **Go to [github.com](https://github.com)**
2. **Click "New repository"** (green button)
3. **Repository name**: `freshtrack-backend`
4. **Description**: `Backend API for FreshTrack food freshness detection app`
5. **Visibility**: Public ✅
6. **Initialize**: ❌ Don't check any boxes
7. **Click "Create repository"**

---

## 🔧 **Step 2: Prepare Your Backend Folder**

### **Navigate to backend directory:**
```bash
cd backend
```

### **Run the deployment helper:**
```bash
npm run deploy:github
```

---

## 📁 **Step 3: Initialize Git Repository**

### **If git is not initialized:**
```bash
git init
```

### **Add all files:**
```bash
git add .
```

### **Check what will be committed:**
```bash
git status
```

---

## 🔗 **Step 4: Connect to GitHub**

### **Add your GitHub remote (replace YOUR_USERNAME):**
```bash
git remote add origin https://github.com/YOUR_USERNAME/freshtrack-backend.git
```

### **Verify remote was added:**
```bash
git remote -v
```

---

## 📤 **Step 5: Push to GitHub**

### **Make your first commit:**
```bash
git commit -m "Initial commit: FreshTrack backend API"
```

### **Set main branch:**
```bash
git branch -M main
```

### **Push to GitHub:**
```bash
git push -u origin main
```

---

## 🚀 **Step 6: Deploy to Railway**

1. **Go to [railway.app](https://railway.app)**
2. **Sign up/Login** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your `freshtrack-backend` repository**
6. **Railway will auto-detect Node.js**
7. **Click "Deploy"**

---

## ⚙️ **Step 7: Configure Railway Environment Variables**

In Railway dashboard, add these variables:

```bash
NODE_ENV=production
PORT=3000
MODEL_PATH=./models/best.onnx
UPLOAD_PATH=./uploads
```

---

## 🔄 **Step 8: Auto-Deployment Setup**

### **Every time you push to GitHub:**
1. Railway detects changes
2. Automatically rebuilds
3. Deploys new version
4. Zero downtime

### **Test auto-deployment:**
```bash
# Make a small change
echo "// Updated at $(date)" >> server.js

# Commit and push
git add .
git commit -m "Test auto-deployment"
git push
```

---

## 📱 **Step 9: Update Your Mobile App**

After Railway deployment, get your backend URL and update:

```typescript
// In services/api.ts
const API_BASE_URL = 'https://your-app.railway.app';
```

---

## 🧪 **Step 10: Test Your Deployment**

### **Test health endpoint:**
```bash
curl https://your-app.railway.app/health
```

### **Test image upload:**
```bash
curl -X POST \
  -F "image=@test-image.jpg" \
  https://your-app.railway.app/api/detect
```

---

## 🚨 **Common Issues & Solutions**

### **Issue: "Repository not found"**
**Solution**: Check repository name and your GitHub username

### **Issue: "Permission denied"**
**Solution**: Make sure repository is public or you have access

### **Issue: "Large file" error**
**Solution**: Check .gitignore excludes large files

### **Issue: Railway build fails**
**Solution**: Check package.json scripts and dependencies

---

## 📊 **File Structure After Deployment**

```
freshtrack-backend/
├── .git/                    # Git repository
├── .gitignore              # Git ignore rules
├── package.json            # Dependencies
├── server.js               # Main server file
├── models/                 # ML models
│   └── best.onnx
├── services/               # Business logic
│   ├── imageProcessor.js
│   └── storageService.js
├── uploads/                # File uploads
│   └── .gitkeep
├── deploy-to-github.js     # Deployment helper
└── GITHUB_DEPLOYMENT.md    # This guide
```

---

## 🎯 **Quick Commands Reference**

```bash
# Initialize and setup
cd backend
npm run deploy:github

# Git commands
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/freshtrack-backend.git
git branch -M main
git push -u origin main

# Update and redeploy
git add .
git commit -m "Update message"
git push
```

---

## 🌟 **Pro Tips**

1. **Always check git status** before committing
2. **Use meaningful commit messages**
3. **Test locally** before pushing
4. **Monitor Railway logs** for errors
5. **Keep .env file local** (never commit secrets)

---

## 🎉 **You're Done!**

Your FreshTrack backend is now:
- ✅ **On GitHub** for version control
- ✅ **Auto-deploying** to Railway
- ✅ **Live and accessible** from anywhere
- ✅ **Ready for your mobile app**

---

**Happy Deploying! 🚀**
