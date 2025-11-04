from datetime import  datetime
def generate_attention_heatmap(text, attention_weights):
    tokens = tokenizer.tokenize(text)[:50]
    attention_array = attention_weights.cpu().detach().numpy()
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_array[:len(tokens), :len(tokens)],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'})
    plt.title('Attention Heatmap - Model Explainability')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = f'/content/attention_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    return heatmap_path
print("Attention visualization function defined!")
