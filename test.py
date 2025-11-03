class DocumentDatasetLoader:
    def __init__(self, cache_dir='./data'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Tesseract is already installed system-wide, no need to initialize
        print("Tesseract OCR ready")

    def load_invoice_dataset(self):
        invoice_dir = os.path.join(self.cache_dir, 'invoices')
        if os.path.exists(invoice_dir) and len(os.listdir(invoice_dir)) > 0:
            print("Dataset already downloaded")
        else:
            try:
                print("Downloading from Kaggle...")
                os.system(f'kaggle datasets download -d urbikn/sroie-datasetv2 -p {self.cache_dir}/')
                print("Extracting files...")
                os.system(f'unzip -q {self.cache_dir}/sroie-datasetv2.zip -d {invoice_dir}/')
                print("Dataset downloaded and extracted")
            except Exception as e:
                print(f"Error downloading: {e}")
                return None
        
        invoices = []
        img_paths = glob.glob(os.path.join(invoice_dir, '**/*.jpg'), recursive=True) + \
                    glob.glob(os.path.join(invoice_dir, '**/*.png'), recursive=True)

        print(f"Found {len(img_paths)} invoice images")

        for img_path in img_paths[:200]:
            try:
                image = Image.open(img_path).convert('RGB')
                txt_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
                text = ""
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                
                if not text:
                    # Use Tesseract OCR instead of EasyOCR
                    text = self.perform_ocr(image)

                invoices.append({
                    'image': image,
                    'text': text,
                    'image_path': img_path,
                    'document_type': 'invoice'
                })

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        print(f"Loaded {len(invoices)} invoices")
        return invoices

    def load_resume_dataset(self):
        resume_dir = os.path.join(self.cache_dir, 'resumes')
        try:
            if os.path.exists(resume_dir):
                print("Loading from disk...")
                dataset = load_from_disk(resume_dir)
            
            print(f"Dataset structure: {dataset.features}")
            print(f"Total resumes: {len(dataset)}")
            resumes = []

            for idx, item in enumerate(dataset):
                if idx >= 200:
                    break

                try:
                    text_parts = []
                    if 'personal_info' in item and item['personal_info']:
                        text_parts.append(f"PERSONAL INFORMATION:\n{item['personal_info']}")
                    if 'experience' in item and item['experience']:
                        text_parts.append(f"\nEXPERIENCE:\n{item['experience']}")
                    if 'education' in item and item['education']:
                        text_parts.append(f"\nEDUCATION:\n{item['education']}")
                    if 'skills' in item and item['skills']:
                        text_parts.append(f"\nSKILLS:\n{item['skills']}")
                    if 'projects' in item and item['projects']:
                        text_parts.append(f"\nPROJECTS:\n{item['projects']}")
                    if 'certifications' in item and item['certifications']:
                        text_parts.append(f"\nCERTIFICATIONS:\n{item['certifications']}")
                    if 'achievements' in item and item['achievements']:
                        text_parts.append(f"\nACHIEVEMENTS:\n{item['achievements']}")
                    
                    full_text = '\n'.join(text_parts)

                    if full_text.strip():
                        image = self._create_text_image(full_text[:1000])
                        resumes.append({
                            'image': image,
                            'text': full_text,
                            'personal_info': item.get('personal_info', ''),
                            'experience': item.get('experience', ''),
                            'education': item.get('education', ''),
                            'skills': item.get('skills', ''),
                            'projects': item.get('projects', ''),
                            'document_type': 'resume'
                        })

                except Exception as e:
                    print(f"Error processing resume {idx}: {e}")
                    continue

            print(f"Loaded {len(resumes)} resumes")
            return resumes

        except Exception as e:
            print(f"Error loading resume dataset: {e}")
            return None

    def load_report_dataset(self):
        print("Loading ML-ArXiv Papers Dataset...")
        papers_dir = os.path.join(self.cache_dir, 'papers')
        try:
            if os.path.exists(papers_dir):
                dataset = load_from_disk(papers_dir)
            else:
                print("Downloading from HuggingFace...")
                dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
                dataset.save_to_disk(papers_dir)

            print(f"Dataset structure: {dataset.features}")
            print(f"Total papers: {len(dataset)}")
            reports = []

            for idx, item in enumerate(dataset):
                if idx >= 200:
                    break

                try:
                    title = item.get('title', 'Untitled')
                    abstract = item.get('abstract', '')
                    full_text = f"TITLE: {title}\n\nABSTRACT:\n{abstract}"

                    if full_text.strip():
                        image = self._create_text_image(full_text[:1000])

                        reports.append({
                            'image': image,
                            'text': full_text,
                            'title': title,
                            'abstract': abstract,
                            'document_type': 'report'
                        })

                except Exception as e:
                    print(f"Error processing paper {idx}: {e}")
                    continue

            print(f"Loaded {len(reports)} research papers")
            return reports

        except Exception as e:
            print(f"Error loading papers dataset: {e}")
            return None

    def _create_text_image(self, text: str, img_size=(800, 600)):
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        margin = 20
        y = margin
        max_width = img_size[0] - 2 * margin

        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines[:40]:
            draw.text((margin, y), line, fill='black', font=font)
            y += 15
            if y > img_size[1] - margin:
                break

        return img

    def perform_ocr(self, image):
        """
        CHANGED: Use Tesseract instead of EasyOCR
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # Tesseract configuration for better document accuracy
        config = '--psm 6 --oem 3'  # PSM 6: single block of text, OEM 3: both legacy and neural
        
        try:
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image, config=config)
            return text
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
            return ""

    def prepare_mixed_dataset(self):
        datasets_dict = {
            'invoice': self.load_invoice_dataset(),
            'resume': self.load_resume_dataset(),
            'report': self.load_report_dataset()
        }
        datasets_dict = {k: v for k, v in datasets_dict.items() if v is not None}

        print(f"\nSuccessfully loaded {len(datasets_dict)} dataset types")
        for doc_type, data in datasets_dict.items():
            if data:
                print(f"{doc_type}: {len(data)} samples")

        return datasets_dict

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,@$%-]', '', text)
        return text.strip()

    def extract_entities(self, text: str) -> Dict:
        doc = self.nlp(text)
        entities = {
            'persons': [],
            'organizations': [],
            'dates': [],
            'money': [],
            'locations': []
        }

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)

        return entities

    def extract_invoice_fields(self, text: str) -> Dict:
        fields = {}
        inv_patterns = [
            r'(?:invoice|inv)[\s#:]*([A-Z0-9\-/]+)',
            r'invoice\s*number[\s:]*([A-Z0-9\-/]+)',
        ]
        for pattern in inv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['invoice_no'] = match.group(1)
                break
        
        vendor_patterns = [
            r'(?:from|vendor|company)[\s:]+([A-Z][A-Za-z\s&]+(?:Ltd|Inc|LLC|Pvt|Corp)?)',
        ]
        for pattern in vendor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['vendor'] = match.group(1).strip()
                break
        
        total_patterns = [
            r'(?:total|amount|grand\s*total)[\s:]*[$₹€£]?\s*([\d,]+\.?\d*)',
            r'[$₹€£]\s*([\d,]+\.?\d*)\s*(?:total)',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['total_amount'] = match.group(1)
                break
        
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                fields['date'] = match.group(0)
                break

        return fields

    def extract_resume_fields(self, text: str) -> Dict:
        fields = {}
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            fields['email'] = email_match.group(0)
        
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                fields['phone'] = match.group(0)
                break
        
        skill_keywords = [
            'python', 'java', 'javascript', 'c\\+\\+', 'c#', 'ruby', 'php', 'swift',
            'machine learning', 'deep learning', 'data science', 'ai', 'artificial intelligence',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
            'react', 'angular', 'vue', 'node', 'django', 'flask',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'agile', 'scrum'
        ]

        found_skills = []
        text_lower = text.lower()
        for skill in skill_keywords:
            if re.search(r'\b' + skill + r'\b', text_lower):
                found_skills.append(skill.title())

        if found_skills:
            fields['skills'] = list(set(found_skills))[:10]
        
        name_match = re.search(r'^([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        if name_match:
            fields['name'] = name_match.group(1)
        
        exp_patterns = [
            r'(\d+)\s*(?:\+)?\s*years?\s*(?:of)?\s*experience',
            r'experience[:\s]+(\d+)\s*years?',
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['experience_years'] = match.group(1)
                break

        return fields

    def extract_report_fields(self, text: str) -> Dict:
        fields = {}
        title_match = re.search(r'(?:TITLE:|Title:)\s*(.+?)(?:\n|ABSTRACT)', text, re.IGNORECASE)
        if title_match:
            fields['title'] = title_match.group(1).strip()
        else:
            first_line = text.split('\n')[0].strip()
            if len(first_line) > 10:
                fields['title'] = first_line
        
        abstract_match = re.search(r'(?:ABSTRACT:|Abstract:)\s*(.+?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            fields['abstract'] = abstract_match.group(1).strip()[:500]
        
        keywords_match = re.search(r'(?:KEYWORDS?:|Keywords?:)\s*(.+?)(?:\n|\Z)', text, re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            keywords = [k.strip() for k in re.split(r'[,;]', keywords_text)]
            fields['keywords'] = keywords[:10]

        return fields
