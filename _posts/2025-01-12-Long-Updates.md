---
title: "Long Updates"
comments : true
share : true
categories:
    - Journal
---
Hello everyone, long time no see. This journal is going to be a little bit long considering there are a lot of gaps that I miss since wednesday. Please bear it with me and I hope I dont forget.

## 2 of My Best Friends Switch Teams

This news came as a shock! My two friends—key players in our team—have transitioned to different departments, and their new roles seem like an odd fit for their passions and expertise. One has joined the support and comprehensive analysis teams, where, despite the overlap in data analytics tools, the teammates there seem... let’s just say, not particularly tech-savvy.

It raises some eyebrows about the political dynamics at play. I have a lot of respect for my team leader—he's incredibly kind, patient, and curious about leveraging data science. But within just a month, we’ve lost two members. It’s unsettling to think that other teams feel entitled to poach our talent, especially when the replacements are not as warmly received. Speaking of which, our newest addition hasn’t exactly won the popularity contest. Let’s just say “needy” might be an understatement.

And here’s where my frustration peaks: my department head. What’s going on with him? He doesn’t engage with the newer members of the analytics team—like we’re invisible or, worse, outsiders. Referring to us as "MT guys" feels dismissive. He chats and collaborates with the senior members but not with us. It’s disheartening and raises serious concerns about the direction of our data analytics department.

## Taking Over Some Jobs (and Bali?)

With my friends gone, their workload didn’t magically disappear—it’s been passed on to the rest of us. And, as you might expect, we’re now scrambling due to a lack of proper knowledge transfer. I don’t mind the added responsibility, but digesting new tasks while covering ongoing projects? That’s a tough balancing act. Below are some notes I’ve been using to keep track of things:

```markdown
# Catatan Proses Otomasi

## Almafact
- *Proses Eksekusi*:  
  Jalankan script: [REDACTED].  
  - *Output*:  
    - skai.almafact_raw  
    - skai.almafact  
- *PR*:  
  Buat job menggunakan akun *cdh audit* setelah mengetahui tanggal update Almafact setiap bulannya.

---

## Tipologi Gaming
1. *Script Generate*  
   - Path: [REDACTED]  
   - *Status*: Sudah job (auto-run setelah TIP 002).  
   - *Catatan*: Karena data Almafact belum pasti tanggal updatenya, bisa saja perlu dijalankan manual nantinya.  

2. *Script Whitelist*  
   - Path: [REDACTED]  
   - *Status*: Sudah job, tetapi bisa jadi run manual tergantung update Almafact.  
   - *Catatan*: Untuk whitelist, tanya [REDACTED].  

3. *Script Pecah 18 Region dan Convert ke CSV*  
   - Path: [REDACTED]  
   - *Status*: Belum job, [REDACTED] akan membuatnya setelah mengetahui tanggal update Almafact.  

4. *Script Convert CSV ke Formatted XLSX*  
   - Path: [REDACTED]  
   - *Status*: Sama dengan nomor 3.  

5. *Script Auto Update Dashboard Bridrive*  
   - Path: [REDACTED]  
   - *Status*:  
     - Script sudah benar tetapi belum teruji.  
     - Jika error, referensi: [REDACTED].  

### PR Tipologi Gaming
- Cari tahu tanggal update data Almafact untuk menyetel job tipologi gaming sesuai tanggal tersebut.
- Tambahkan fitur *auto-upload* .XLSX ke Bridrive ([REDACTED] perlu menambahkannya, karena sebelumnya diupload manual).

---

## SCORING_MONBER Recovery (Versi 3)
1. *Script Generate*  
   - Path: [REDACTED]  
   - Jalankan *Part 1* dan *Part 2*.  
   - *Output*:  
     - skai.monber_scoring_{tahun}{bulan}  
     - *Jika mengikuti format RAO*: skai.monber_score_{tahun}{bulan}.  

2. *PR*:  
   - Belum ada script untuk auto pecah ke 18 regional.  
   - Belum ada script untuk auto update ke TWBX.  

3. *Format TWBX*:  
   - Gunakan referensi dari [REDACTED].  

4. *Catatan*:  
   - Scoring Monber versi 3 *belum resmi dipakai*.  
   - Rencana resmi digunakan mulai Februari setelah sosialisasi ke RAO (hubungi [REDACTED] untuk info lebih lanjut).  

5. *Status Job*:  
   - Belum dibuat.  
   - Rencana dibuat setelah resmi digunakan.  
   - Job akan menggunakan akun *cdh audit*.  


```

On a brighter note, my network analysis project is running smoothly. No additional tweaks needed for now.

Oh, and guess what? I might be heading to Bali soon for a sharing session on data analytical tools! While the idea of presenting excites me, I’m also pretty nervous. What should I prepare? Even my accommodation details are still up in the air. One step at a time, right?

## Dating: OMG She’s the Best ❤️

Holy moly, where do I even start? This date was beyond amazing. I didn’t expect to feel so much joy—cuddling, holding hands... it was like a dream. (Note to self: hope she never reads this!)

This girl is everything. She’s beautiful, yes, but also warm, funny, and easy to talk to. We shared stories, laughed, and explored so many little details about each other. The best moment? When she got sleepy, rested her head on my shoulder, and held my hand. I’m smitten.

That said, I was a bit nervous. I wanted to hug her, maybe even kiss her, but I held back—waiting for the right moment when she feels more comfortable.

Here’s a funny twist: I think my "liquid luck" (zam-zam water) worked! Even though I messed up navigating our route home (oops), the date ended perfectly. Next up? An archery date and coffee outing!

One little worry, though—I’m nervous about discussing finances with her. I want to treat her, but this month’s budget is tight, especially with the Bali trip looming. I hope she understands when the time comes. Fingers crossed!

> "Love on the first date isn’t just about sparks; it’s the comfort of shared smiles, the magic of effortless conversation, and the quiet hope that this is the beginning of something beautiful."
>
> -- anonymous
