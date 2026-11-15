---
title: "The Complete Beginner’s Guide to Vercel: Put Your Website Online in Minutes"
lang: en
takeaways:
  - "What is Vercel and basic way to use vercel"
header:
  teaser: https://vercel.com/geist/brands/assets/vercel-logotype-white.svg
comments: true
share: true
categories:
  - Tips
tags:
  - Vercel
---

# *Turn code sitting on your laptop into a fast, live website with a real link in under 60 seconds.*

Have you ever built a cool website on your laptop and wondered: *"How do I get a real link so I can share this with my friends, family, or future employers?"*

In the old days, putting a website on the internet was hard. You had to rent a complicated computer in a data center, set up security certificates by hand, and manually upload files every time you fixed a typo.

**Vercel** makes all of that disappear. It takes the code sitting on your computer and turns it into a fast, live website with a real link—often in less than 60 seconds.

In this guide, we will explain everything in plain English and walk through real, step-by-step examples with visual diagrams.

## 1. What is Vercel? (In Plain English)

![Global Edge Network](https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1200&q=80&auto=format&fit=crop)
*Vercel copies your website across a global network so visitors anywhere in the world load it instantly.*

Think of **Vercel** as a **magic publishing house for websites**. You hand Vercel your raw code files, and it automatically packs them up, puts them on super-fast computers around the world, and hands you back a live website link.

### How Vercel Works (At a Glance)

```
+-------------------+       +-------------------+       +-------------------+       +-------------------+
|   Your Laptop     |       |      GitHub       |       |      Vercel       |       |   Live Website    |
| (Write your code) | ----> | (Save your code)  | ----> | (Build & Publish) | ----> | (your-site.app)   |
+-------------------+       +-------------------+       +-------------------+       +-------------------+
```

### Why do beginners love it?

- **No Setup Headaches:** Whether you wrote plain HTML, React, Next.js, or Vue, Vercel looks at your files and automatically figures out how to turn them into a website.
- **Automatic Updates:** Once you connect Vercel to GitHub, every time you save and upload new code to GitHub, your live website updates itself automatically.
- **Super Fast Everywhere (Global CDN):** Imagine if a pizza chain only had one kitchen in New York—customers in Tokyo would wait forever for delivery. Vercel copies your website to data centers all over the globe so it loads instantly for everyone:

```
                       [ Vercel Master Copy ]
                       /         |          \
                      v          v           v
              [ New York ]   [ London ]   [ Tokyo ]
                   |             |            |
                   v             v            v
               Visitor A     Visitor B    Visitor C
              (Loads fast!) (Loads fast!) (Loads fast!)
```

- **Free Security Padlock (HTTPS):** Vercel automatically gives your site a free SSL security certificate so browsers don't show scary "Not Secure" warnings.
- **100% Free to Start:** Vercel’s "Hobby" plan is completely free for personal projects, portfolios, and learning.

---

## 2. 4 Simple Words You Need to Know

Before we touch any buttons, let's translate four tech words into everyday concepts:

1. **Production (Your "Open for Business" Site):**
   This is your main, official website link (like `my-cool-site.vercel.app`) that you share with the world.
2. **Preview Deployment (Your "Fitting Room" Link):**
   Want to test a brand-new design without breaking your real website? When you test changes on a separate branch in GitHub, Vercel creates a secret, temporary link just for testing.

```
GitHub Code
 ├── main branch ---------> Production URL (my-site.vercel.app)       [Public & Official]
 └── new-design branch ---> Preview URL    (my-site-git-new.vercel.app) [Private Testing Link]
```

1. **Serverless Functions (Your "Vending Machine" Backend):**
   Normally, running backend code requires a computer running 24/7. Serverless functions are like a vending machine: they sleep and cost nothing until a visitor clicks a button, wake up for half a second to do a task (like sending an email or checking the weather), and go right back to sleep.
2. **Environment Variables (Your "Locked Safe"):**
   A secure vault inside Vercel where you hide secret passwords and API keys so hackers can't steal them from your public code.

---

## 3. What You Need Before Starting

To follow along with this guide, grab these free tools:

```
  [1. GitHub Account]   +   [2. Vercel Account]   +   [3. Node.js (Optional)]
  (Stores your code)        (Publishes your site)     (For React & Terminal)
```

- **A GitHub Account:** Think of GitHub as *Google Drive for code*. Sign up for free at [github.com](https://github.com).
- **A Vercel Account:** Go to [vercel.com](https://vercel.com) and click **"Sign Up"**. Choose **"Continue with GitHub"** so the two accounts link together automatically.
- **Node.js (Optional):** Only needed if you plan to build React apps or use terminal commands. You can download it at [nodejs.org](https://nodejs.org).

---

## 4. Method 1: The Point-and-Click Way (Using the Website)

If you don't like typing commands into a black terminal screen, this method is for you!

### Visual Walkthrough

```
[Step 1: GitHub]             [Step 2: Vercel Dashboard]       [Step 3: Live!]
Upload your files  ------->  Click "Add New -> Project"  --->  Click "Deploy"
to a repository              and pick your repo               (Confetti & Live Link!)
```

### Step 1: Put your code on GitHub

Create a new repository on GitHub and upload your website files there.

### Step 2: Import your project into Vercel

1. Open your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click the **"Add New..."** button and pick **"Project"**.
3. You will see a list of your GitHub folders (repositories). Find your website and click **"Import"**. *(Don't see it? Click "Adjust GitHub App Permissions" to give Vercel permission to view it).*

### Step 3: Hit Deploy!

1. **Project Name:** Type a name for your site. This becomes your free web address (`your-name.vercel.app`).
2. **Settings:** Leave all the build settings just as they are—Vercel already guessed them for you!
3. Click the big **"Deploy"** button.

Wait about 30 to 60 seconds. You’ll see virtual confetti celebrate on your screen! Click the screenshot of your site to open your brand-new live URL.

---

## 5. Method 2: The Fast Terminal Way (Using Vercel CLI)

![Developer Terminal Workflow](https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1200&q=80&auto=format&fit=crop)
*Using the Vercel CLI lets you push code directly from your computer's terminal to the cloud in seconds.*

Want to upload a folder straight from your computer without creating a GitHub repository first? You can use the **Vercel CLI** (a small helper tool you run in your computer's terminal).

```
Your Computer Folder  ---( vercel )--->  Preview URL (Test it first!)
Your Computer Folder  -( vercel --prod )-> Production URL (Live for everyone!)
```

### Step 1: Install the helper tool

Open your terminal (Command Prompt on Windows, Terminal on Mac) and type:

```bash
npm install -g vercel
```

### Step 2: Log in

Type this command and press Enter. It will open your web browser so you can click "Confirm" to log in:

```bash
vercel login
```

### Step 3: Upload a test version

Go inside your website's folder in the terminal and type `vercel`:

```bash
cd my-website-folder
vercel
```

Vercel will ask a few quick yes/no questions—just press **Enter** to accept the default answers for all of them. In seconds, it will hand you a **Preview Link** to test your site.

### Step 4: Make it official!

Happy with how it looks? Type this command to push it to your main public link:

```bash
vercel --prod
```

---

## 6. 3 Hands-On Examples You Can Copy & Try

Let’s look at three real projects—starting from a basic webpage and working up to a full app with a backend.

### Example 1: A Simple HTML, CSS & JavaScript Page

You don't need fancy frameworks to use Vercel. Imagine you have a folder called `my-portfolio` with three files:

```
my-portfolio/
├── index.html   <-- Vercel automatically loads this as your homepage!
├── style.css
└── script.js
```

**`index.html`:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My First Website</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <h1>Welcome to My Website!</h1>
  <button id="btn">Say Hello</button>
  <script src="script.js"></script>
</body>
</html>
```

**How to put it online:**

1. Open your terminal inside the `my-portfolio` folder.
2. Run `vercel --prod`.
3. Done! Vercel automatically shows `index.html` as your front page.

---

### Example 2: A React App (Built with Vite)

React is a popular tool for building interactive web apps. Let's create one and put it on Vercel.

**Step 1: Create your React app on your computer**

```bash
npm create vite@latest my-react-app -- --template react
cd my-react-app
npm install
```

**Step 2: Prevent the "404 Page Not Found" Error (Super Important!)**
If your React app has multiple pages (like `/about` or `/contact`), clicking the browser's **Refresh** button on Vercel can sometimes cause a `404: NOT_FOUND` error. Here is why—and how `vercel.json` fixes it:

```
WITHOUT vercel.json:
Visitor refreshes "/about" ---> Vercel looks for "about.html" ---> 404 NOT FOUND!

WITH vercel.json:
Visitor refreshes "/about" ---> Vercel loads "index.html" ---> React shows About Page!
```

To fix this forever, create a tiny file named `vercel.json` in your main project folder and paste this inside:

**`vercel.json`:**

```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

*(In plain English, this tells Vercel: "No matter what page link someone visits, always load `index.html` and let React handle the rest!")*

**Step 3: Deploy**
Upload your folder to GitHub and import it into Vercel (using Method 1). Vercel will automatically build the app and publish it!

---

### Example 3: Adding a Mini Backend (Serverless API)

What if your website needs to run secret code on a server, like fetching data or generating a custom greeting? With Vercel, you just create a folder named `api` and put a JavaScript file inside it!

**How Frontend & Backend Talk on Vercel:**

```
+--------------------+   1. Asks /api/greet?name=Alex   +----------------------+
|    index.html      | -------------------------------> |    api/greet.js      |
| (Visitor's Screen) | <------------------------------- | (Vercel Serverless)  |
+--------------------+   2. Replies with JSON message   +----------------------+
```

**Folder Structure:**

```
my-fullstack-app/
├── index.html
└── api/
    └── greet.js    <-- Automatically becomes your-site.vercel.app/api/greet
```

**`api/greet.js` (Your backend code):**

```javascript
export default function handler(request, response) {
  // Grab the visitor's name from the link, or use "Friend" if none was given
  const { name = 'Friend' } = request.query;

  // Send back a message and the current server time
  response.status(200).json({
    message: `Hello, ${name}! Welcome to Vercel.`,
    time: new Date().toLocaleTimeString(),
  });
}
```

**`index.html` (Your webpage asking the backend for data):**

```html
<!DOCTYPE html>
<html lang="en">
<body>
  <h2 id="greeting">Loading message...</h2>

  <script>
    // Ask our /api/greet backend for a message for "Alex"
    fetch('/api/greet?name=Alex')
      .then(res => res.json())
      .then(data => {
        document.getElementById('greeting').innerText = 
          `${data.message} (Loaded at ${data.time})`;
      });
  </script>
</body>
</html>
```

When you deploy this, Vercel automatically turns `api/greet.js` into a live backend link (`your-site.vercel.app/api/greet`). No server setup required!

---

## 7. Keeping Your Passwords & API Keys Safe

![Digital Security Vault](https://images.unsplash.com/photo-1578930955087-58a5d98e7a95?w=1200&q=80&auto=format&fit=crop)
*Store sensitive API keys inside Vercel's encrypted Environment Variables instead of pasting them into public GitHub files.*

Never paste secret API keys (like OpenAI, Stripe, or database passwords) directly into your code files. If you upload them to GitHub, anyone can see them!

```
BAD:   Code File  ---> "API_KEY = 12345" ---> Uploaded to GitHub ---> Anyone can steal it!
GOOD:  Code File  ---> "process.env.API_KEY" ---> Reads from Vercel's Encrypted Safe!
```

Instead, put them in Vercel's **Locked Safe (Environment Variables)**:

1. Open your project on the **Vercel Dashboard**.
2. Click **Settings** at the top, then click **Environment Variables** on the left.
3. Type a **Name** (for example: `WEATHER_API_KEY`) and paste your secret **Value** (for example: `secret12345`).
4. Click **Save**.

> **Important Tip:** After adding or changing a secret key, go to the **Deployments** tab, click the three dots (`...`) next to your latest deployment, and click **Redeploy** so your website picks up the new key!

### How to use that secret key in your code:

- **In your `/api` backend files:** Write `process.env.WEATHER_API_KEY`
- **In a Vite / React frontend:** Start the name with `VITE_` (like `VITE_WEATHER_API_KEY`) and write `import.meta.env.VITE_WEATHER_API_KEY`

---

## 8. Using Your Own Custom Domain Name

Right now, your website ends in `.vercel.app`. If you buy your own domain name (like `yourname.com` from Namecheap, GoDaddy, or Cloudflare), connecting it takes two minutes:

```
Visitor types "yourname.com"
          |
          v
[ Domain Registrar DNS ]  ---> Points to Vercel (76.76.21.21)
          |
          v
[ Your Vercel Website ]   ---> Opens with free HTTPS Padlock!
```

1. In your Vercel project, go to **Settings** -> **Domains**.
2. Type in `yourname.com` and click **Add**.
3. Vercel will show you two short codes (called **DNS Records**):
   - An **A Record** pointing to `76.76.21.21`
   - A **CNAME Record** pointing to `cname.vercel-dns.com`
4. Log in to the website where you bought your domain, go to its **DNS Settings**, and paste those two records in.

Once the internet updates (usually within 10 to 30 minutes), typing `yourname.com` will open your Vercel website with a secure HTTPS padlock!

---

## 9. Common Beginner Hiccups & Easy Fixes

Even pros run into small bumps. Here is how to fix the four most common beginner issues:

| What Happened? | Why It Happened | How to Fix It |
| --- | --- | --- |
| **It works on my laptop, but fails to build on Vercel!** | **Capitalization mismatch.** Windows and Mac don't care if you write `import Header from './header.js'` when the file is `Header.js`. Vercel's servers are strict! | Make sure the uppercase and lowercase letters in your `import` lines match your actual filenames 100%. |
| **I get a "404 Not Found" error when refreshing a page.** | You are using React or Vue with multiple pages, and Vercel doesn't know how to route subpages. | Add the `vercel.json` rewrite file shown in **Example 2** above. |
| **My API key says `undefined`.** | You forgot to redeploy after saving the key in Vercel, or forgot the `VITE_` prefix. | Click **Redeploy** in your Vercel Deployments tab, and check that frontend keys start with `VITE_`. |
| **I uploaded new code, but the live site didn't change.** | You uploaded your code to a test branch on GitHub instead of the `main` branch. | Check the **Deployments** tab in Vercel—your changes are waiting for you on a temporary **Preview URL**! |

---

## You're Ready to Launch!

Putting a website on the internet used to be one of the hardest parts of coding. With Vercel, it is one of the easiest.

**Your Challenge:** Grab a small HTML or React folder on your computer right now, upload it to GitHub, and deploy your first live link on Vercel today!
