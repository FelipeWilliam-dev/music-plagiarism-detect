#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de Banco de Dados de Similaridade Musical (Versão Avançada)

Este script foi melhorado para:
- Extrair múltiplas características musicais usando librosa
- Apresentar compatibilidade de cada característica
- Balancear automaticamente as classes (reduzindo a maior)
- Análise detalhada de performance das características
"""

# --- 1. Importação das Bibliotecas ---
import os
import numpy as np
import pandas as pd
import librosa
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from librosa.sequence import dtw
from pydub import AudioSegment
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random

warnings.filterwarnings('ignore')

# --- 2. Configuração de Características ---

CARACTERISTICAS_CONFIG = {
    'mfcc': {
        'nome': 'MFCC (Mel-Frequency Cepstral Coefficients)',
        'descricao': 'Representa características espectrais e timbre',
        'parametros': {'n_mfcc': 13},
        'funcao': lambda y, sr: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
        'ativo': True
    },
    'chroma': {
        'nome': 'Chroma Features',
        'descricao': 'Representa conteúdo harmônico e tonalidade',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.chroma_stft(y=y, sr=sr),
        'ativo': True
    },
    'spectral_centroid': {
        'nome': 'Spectral Centroid',
        'descricao': 'Centro de massa do espectro (brilho)',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.spectral_centroid(y=y, sr=sr),
        'ativo': True
    },
    'spectral_bandwidth': {
        'nome': 'Spectral Bandwidth',
        'descricao': 'Largura do espectro',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.spectral_bandwidth(y=y, sr=sr),
        'ativo': True
    },
    'spectral_rolloff': {
        'nome': 'Spectral Roll-off',
        'descricao': 'Frequência onde 85% da energia está concentrada',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.spectral_rolloff(y=y, sr=sr),
        'ativo': True
    },
    'zero_crossing_rate': {
        'nome': 'Zero Crossing Rate',
        'descricao': 'Taxa de cruzamento por zero (textura)',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.zero_crossing_rate(y),
        'ativo': True
    },
    'tempo': {
        'nome': 'Tempo',
        'descricao': 'Velocidade da música (BPM)',
        'parametros': {},
        'funcao': lambda y, sr: np.array([[librosa.beat.tempo(y=y, sr=sr)[0]]]),
        'ativo': True
    },
    'rms': {
        'nome': 'RMS Energy',
        'descricao': 'Energia RMS (volume/intensidade)',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.rms(y=y),
        'ativo': True
    },
    'mel_spectrogram': {
        'nome': 'Mel Spectrogram',
        'descricao': 'Representação mel-scale do espectro',
        'parametros': {'n_mels': 128},
        'funcao': lambda y, sr: librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128),
        'ativo': True
    },
    'tonnetz': {
        'nome': 'Tonnetz',
        'descricao': 'Representação harmônica tonal',
        'parametros': {},
        'funcao': lambda y, sr: librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr),
        'ativo': True
    }
}

# --- 3. Funções de Validação e Conversão ---

def validar_arquivo_audio(caminho_arquivo):
    """Valida se um arquivo de áudio é válido e pode ser processado."""
    try:
        if not os.path.exists(caminho_arquivo):
            return False
        if os.path.getsize(caminho_arquivo) < 1024:
            return False
        y, sr = librosa.load(caminho_arquivo, duration=1.0)
        if len(y) == 0:
            return False
        return True
    except Exception:
        return False

def converter_mp3_para_wav(caminho_mp3):
    """Converte um arquivo MP3 para WAV, se necessário."""
    if not caminho_mp3.lower().endswith('.mp3'):
        return caminho_mp3

    if not validar_arquivo_audio(caminho_mp3):
        print(f"  ERRO: Arquivo MP3 inválido: {os.path.basename(caminho_mp3)}")
        return None

    wav_path = os.path.splitext(caminho_mp3)[0] + '.wav'
    try:
        audio = AudioSegment.from_mp3(caminho_mp3)
        if len(audio) == 0:
            return None
        audio.export(wav_path, format='wav')
        if not validar_arquivo_audio(wav_path):
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return None
        return wav_path
    except Exception as e:
        print(f"  ERRO ao converter {os.path.basename(caminho_mp3)}: {str(e)}")
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except:
                pass
        return None

# --- 4. Extração de Características Avançada ---

def extrair_todas_caracteristicas(caminho_audio):
    """
    Extrai todas as características configuradas de um arquivo de áudio.
    Retorna um dicionário com as características e seus status.
    """
    try:
        y, sr = librosa.load(caminho_audio, duration=60.0, sr=None)
        
        if len(y) == 0:
            return None, "Arquivo vazio após carregamento"
            
        # Normaliza o áudio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            return None, "Arquivo contém apenas silêncio"
        
        caracteristicas = {}
        status_extracao = {}
        
        for nome_carac, config in CARACTERISTICAS_CONFIG.items():
            if not config['ativo']:
                continue
                
            try:
                feature = config['funcao'](y, sr)
                
                # Verifica se a característica foi extraída corretamente
                if feature is not None and feature.size > 0:
                    # Reduz a dimensionalidade se necessário (pega estatísticas)
                    if feature.ndim > 1:
                        feature_stats = np.concatenate([
                            np.mean(feature, axis=1),
                            np.std(feature, axis=1),
                            np.median(feature, axis=1),
                            np.min(feature, axis=1),
                            np.max(feature, axis=1)
                        ])
                    else:
                        feature_stats = np.array([
                            np.mean(feature),
                            np.std(feature),
                            np.median(feature),
                            np.min(feature),
                            np.max(feature)
                        ])
                    
                    caracteristicas[nome_carac] = feature_stats
                    status_extracao[nome_carac] = "Sucesso"
                else:
                    status_extracao[nome_carac] = "Falha - Feature vazia"
                    
            except Exception as e:
                status_extracao[nome_carac] = f"Erro: {str(e)[:50]}"
        
        return caracteristicas, status_extracao
        
    except Exception as e:
        return None, f"Erro no carregamento: {str(e)}"

def calcular_similaridade_multipla(carac_a, carac_b):
    """
    Calcula similaridade usando múltiplas características.
    Retorna um dicionário com similaridades individuais e combinada.
    """
    similaridades = {}
    
    # Características comuns entre os dois áudios
    carac_comuns = set(carac_a.keys()) & set(carac_b.keys())
    
    if not carac_comuns:
        return None, "Nenhuma característica comum"
    
    # Calcula similaridade para cada característica
    for carac in carac_comuns:
        try:
            # Similaridade coseno
            dot_product = np.dot(carac_a[carac], carac_b[carac])
            norm_a = np.linalg.norm(carac_a[carac])
            norm_b = np.linalg.norm(carac_b[carac])
            
            if norm_a == 0 or norm_b == 0:
                similaridade = 0.0
            else:
                similaridade = dot_product / (norm_a * norm_b)
            
            # Converte para escala 0-1
            similaridades[carac] = (similaridade + 1) / 2
            
        except Exception as e:
            print(f"    Erro ao calcular similaridade para {carac}: {e}")
            continue
    
    if not similaridades:
        return None, "Falha no cálculo de todas as similaridades"
    
    # Similaridade combinada (média ponderada)
    pesos = {
        'mfcc': 0.25,
        'chroma': 0.20,
        'spectral_centroid': 0.15,
        'spectral_bandwidth': 0.10,
        'spectral_rolloff': 0.10,
        'zero_crossing_rate': 0.05,
        'tempo': 0.05,
        'rms': 0.05,
        'mel_spectrogram': 0.03,
        'tonnetz': 0.02
    }
    
    similaridade_combinada = 0.0
    peso_total = 0.0
    
    for carac, sim in similaridades.items():
        peso = pesos.get(carac, 0.1)
        similaridade_combinada += sim * peso
        peso_total += peso
    
    if peso_total > 0:
        similaridade_combinada /= peso_total
    
    similaridades['combinada'] = similaridade_combinada
    
    return similaridades, "Sucesso"

# --- 5. Balanceamento de Classes ---

def balancear_resultados(resultados):
    """
    Balanceia as classes reduzindo amostras da classe majoritária.
    """
    if not resultados:
        return resultados
    
    # Separa por classe
    plagios = [r for r in resultados if r['Plagio'] == 'sim']
    variadas = [r for r in resultados if r['Plagio'] == 'não']
    
    print(f"\nBALANCEAMENTO DE CLASSES:")
    print(f"  Plágios: {len(plagios)} amostras")
    print(f"  Variadas: {len(variadas)} amostras")
    
    # Determina o tamanho da classe minoritária
    min_size = min(len(plagios), len(variadas))
    
    if min_size == 0:
        print("  AVISO: Uma das classes está vazia. Retornando sem balanceamento.")
        return resultados
    
    # Reduz aleatoriamente a classe majoritária
    if len(plagios) > min_size:
        random.shuffle(plagios)
        plagios = plagios[:min_size]
        print(f"  Reduzindo plágios para {min_size} amostras")
    
    if len(variadas) > min_size:
        random.shuffle(variadas)
        variadas = variadas[:min_size]
        print(f"  Reduzindo variadas para {min_size} amostras")
    
    # Combina e embaralha
    resultados_balanceados = plagios + variadas
    random.shuffle(resultados_balanceados)
    
    print(f"  Dataset balanceado: {len(resultados_balanceados)} amostras totais")
    print(f"  Distribuição: {len(plagios)} plágios, {len(variadas)} variadas")
    
    return resultados_balanceados

# --- 6. Análise de Compatibilidade ---

def analisar_compatibilidade_caracteristicas(resultados):
    """
    Analisa a compatibilidade e performance de cada característica.
    """
    if not resultados:
        return
    
    print(f"\n{'='*80}")
    print("ANÁLISE DE COMPATIBILIDADE DAS CARACTERÍSTICAS")
    print(f"{'='*80}")
    
    # Coleta dados de todas as características
    todas_caracteristicas = set()
    for resultado in resultados:
        if 'caracteristicas_individuais' in resultado:
            todas_caracteristicas.update(resultado['caracteristicas_individuais'].keys())
    
    print(f"Características encontradas: {len(todas_caracteristicas)}")
    print(f"Características: {sorted(todas_caracteristicas)}")
    
    # Análise de compatibilidade por característica
    compatibilidade_stats = {}
    
    for carac in todas_caracteristicas:
        valores_plagio = []
        valores_variadas = []
        
        for resultado in resultados:
            if 'caracteristicas_individuais' in resultado and carac in resultado['caracteristicas_individuais']:
                valor = float(resultado['caracteristicas_individuais'][carac])
                if resultado['Plagio'] == 'sim':
                    valores_plagio.append(valor)
                else:
                    valores_variadas.append(valor)
        
        if len(valores_plagio) > 1 and len(valores_variadas) > 1:
            # Estatísticas descritivas
            stats = {
                'total_amostras': len(valores_plagio) + len(valores_variadas),
                'amostras_plagio': len(valores_plagio),
                'amostras_variadas': len(valores_variadas),
                'media_plagio': np.mean(valores_plagio),
                'media_variadas': np.mean(valores_variadas),
                'std_plagio': np.std(valores_plagio),
                'std_variadas': np.std(valores_variadas),
                'diferenca_medias': abs(np.mean(valores_plagio) - np.mean(valores_variadas))
            }
            
            # Correlação com classe (usando valores numéricos)
            todas_vals = valores_plagio + valores_variadas
            labels = [1] * len(valores_plagio) + [0] * len(valores_variadas)
            
            try:
                corr_pearson, p_pearson = pearsonr(todas_vals, labels)
                stats['correlacao_pearson'] = corr_pearson
                stats['p_value_pearson'] = p_pearson
            except:
                stats['correlacao_pearson'] = 0
                stats['p_value_pearson'] = 1
            
            # Poder discriminativo (diferença normalizada)
            std_pooled = np.sqrt((stats['std_plagio']**2 + stats['std_variadas']**2) / 2)
            if std_pooled > 0:
                stats['poder_discriminativo'] = stats['diferenca_medias'] / std_pooled
            else:
                stats['poder_discriminativo'] = 0
                
            compatibilidade_stats[carac] = stats
    
    # Exibe resultados ordenados por poder discriminativo
    print(f"\n{'='*80}")
    print("RANKING DE CARACTERÍSTICAS POR PODER DISCRIMINATIVO")
    print(f"{'='*80}")
    
    ranking = sorted(compatibilidade_stats.items(), 
                    key=lambda x: x[1]['poder_discriminativo'], 
                    reverse=True)
    
    print(f"{'Característica':<20} {'Poder':<8} {'Correlação':<12} {'P-value':<10} {'Amostras':<10}")
    print("-" * 70)
    
    for carac, stats in ranking:
        nome_curto = carac.replace('spectral_', 's_')[:19]
        print(f"{nome_curto:<20} "
              f"{stats['poder_discriminativo']:<8.3f} "
              f"{stats['correlacao_pearson']:<12.3f} "
              f"{stats['p_value_pearson']:<10.3f} "
              f"{stats['total_amostras']:<10}")
    
    # Detalhes das top 5 características
    print(f"\n{'='*80}")
    print("DETALHES DAS TOP 5 CARACTERÍSTICAS")
    print(f"{'='*80}")
    
    for i, (carac, stats) in enumerate(ranking[:5]):
        config = CARACTERISTICAS_CONFIG.get(carac, {})
        print(f"\n{i+1}. {config.get('nome', carac.upper())}")
        print(f"   Descrição: {config.get('descricao', 'N/A')}")
        print(f"   Poder Discriminativo: {stats['poder_discriminativo']:.3f}")
        print(f"   Correlação com Plágio: {stats['correlacao_pearson']:.3f} (p={stats['p_value_pearson']:.3f})")
        print(f"   Média Plágios: {stats['media_plagio']:.3f} ± {stats['std_plagio']:.3f}")
        print(f"   Média Variadas: {stats['media_variadas']:.3f} ± {stats['std_variadas']:.3f}")
        print(f"   Diferença de Médias: {stats['diferenca_medias']:.3f}")
        print(f"   Amostras: {stats['amostras_plagio']} plágios, {stats['amostras_variadas']} variadas")

# --- 7. Processamento de Diretórios ---

def processar_diretorio(diretorio_base, nome_diretorio, eh_plagio):
    """Processa um diretório específico com extração de múltiplas características."""
    diretorio_completo = os.path.join(diretorio_base, nome_diretorio)
    resultados = []
    
    if not os.path.exists(diretorio_completo):
        print(f"AVISO: Diretório '{diretorio_completo}' não encontrado.")
        return resultados
    
    try:
        todos_os_itens = os.listdir(diretorio_completo)
        extensoes_audio = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus')
        
        pastas_validas = []
        for item in todos_os_itens:
            caminho_item = os.path.join(diretorio_completo, item)
            if os.path.isdir(caminho_item):
                try:
                    arquivos_audio = [f for f in os.listdir(caminho_item) 
                                    if f.lower().endswith(extensoes_audio)]
                    
                    arquivos_validos = []
                    for arquivo in arquivos_audio:
                        caminho_completo = os.path.join(caminho_item, arquivo)
                        if validar_arquivo_audio(caminho_completo):
                            arquivos_validos.append(arquivo)
                    
                    if len(arquivos_validos) >= 2:
                        pastas_validas.append(item)
                except Exception:
                    continue
        
        def ordenar_pasta(nome_pasta):
            try:
                return (0, int(nome_pasta))
            except ValueError:
                return (1, nome_pasta.lower())
        
        pastas_validas.sort(key=ordenar_pasta)
        
    except Exception as e:
        print(f"ERRO: Problema ao acessar '{diretorio_completo}': {e}")
        return resultados

    tipo_conteudo = "PLÁGIOS" if eh_plagio else "VARIADAS"
    classe_plagio = "sim" if eh_plagio else "não"
    
    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {tipo_conteudo}")
    print(f"Pastas válidas: {len(pastas_validas)}")
    print(f"{'='*60}")

    pares_processados = 0
    pares_com_erro = 0
    stats_caracteristicas = {}

    for nome_pasta in pastas_validas:
        caminho_da_pasta = os.path.join(diretorio_completo, nome_pasta)
        
        extensoes_audio = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus')
        todos_arquivos_audio = [f for f in os.listdir(caminho_da_pasta) 
                               if f.lower().endswith(extensoes_audio)]
        
        arquivos_de_audio = []
        for arquivo in todos_arquivos_audio:
            caminho_completo = os.path.join(caminho_da_pasta, arquivo)
            if validar_arquivo_audio(caminho_completo):
                arquivos_de_audio.append(arquivo)
        
        if len(arquivos_de_audio) < 2:
            print(f"  ERRO: {nome_pasta} - arquivos insuficientes")
            pares_com_erro += 1
            continue
            
        arquivos_de_audio.sort()
        if len(arquivos_de_audio) > 2:
            arquivos_de_audio = arquivos_de_audio[:2]

        print(f"\nProcessando: {nome_diretorio}/{nome_pasta}")
        print(f"  Arquivos: {arquivos_de_audio[0]} <-> {arquivos_de_audio[1]}")

        caminho_a_original = os.path.join(caminho_da_pasta, arquivos_de_audio[0])
        caminho_b_original = os.path.join(caminho_da_pasta, arquivos_de_audio[1])

        arquivos_temporarios = []

        try:
            print("  Convertendo para WAV...")
            caminho_a_wav = converter_mp3_para_wav(caminho_a_original)
            caminho_b_wav = converter_mp3_para_wav(caminho_b_original)

            if caminho_a_wav != caminho_a_original:
                arquivos_temporarios.append(caminho_a_wav)
            if caminho_b_wav != caminho_b_original:
                arquivos_temporarios.append(caminho_b_wav)

            if not caminho_a_wav or not caminho_b_wav:
                print(f"  Conversão falhou")
                pares_com_erro += 1
                continue

            print("  Extraindo características...")
            carac_a, status_a = extrair_todas_caracteristicas(caminho_a_wav)
            carac_b, status_b = extrair_todas_caracteristicas(caminho_b_wav)

            if carac_a is None or carac_b is None:
                print(f"  Extração de características falhou")
                pares_com_erro += 1
                continue

            # Atualiza estatísticas de extração
            for carac, status in status_a.items():
                if carac not in stats_caracteristicas:
                    stats_caracteristicas[carac] = {'sucesso': 0, 'total': 0}
                stats_caracteristicas[carac]['total'] += 1
                if status == "Sucesso":
                    stats_caracteristicas[carac]['sucesso'] += 1

            print("  Calculando similaridades...")
            similaridades, status_sim = calcular_similaridade_multipla(carac_a, carac_b)

            if similaridades is None:
                print(f"  Cálculo de similaridade falhou: {status_sim}")
                pares_com_erro += 1
                continue

            similaridade_principal = similaridades.get('combinada', 0.0)
            print(f"  Similaridade principal: {similaridade_principal:.2%}")

            # Mostra top 3 características individuais
            sim_individuais = {k: v for k, v in similaridades.items() if k != 'combinada'}
            top_3 = sorted(sim_individuais.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top características: {', '.join([f'{k}={v:.3f}' for k, v in top_3])}")

            id_par = f"{nome_diretorio}_{nome_pasta}"

            resultado = {
                'ID_do_Par': id_par,
                'Tipo_Diretorio': nome_diretorio,
                'Pasta_Original': nome_pasta,
                'Musica_1': arquivos_de_audio[0],
                'Musica_2': arquivos_de_audio[1],
                'Similaridade': f"{similaridade_principal:.4f}",
                'Plagio': classe_plagio,
                'caracteristicas_individuais': {k: f"{v:.4f}" for k, v in sim_individuais.items()}
            }

            # Adiciona características individuais como colunas separadas
            for carac, valor in sim_individuais.items():
                resultado[f'sim_{carac}'] = f"{valor:.4f}"

            resultados.append(resultado)
            pares_processados += 1

        except Exception as e:
            print(f"  ERRO: {e}")
            pares_com_erro += 1

        finally:
            # Limpa arquivos temporários
            for arquivo in arquivos_temporarios:
                try:
                    if os.path.exists(arquivo):
                        os.remove(arquivo)
                except:
                    pass

    # Relatório de características extraídas
    print(f"\n{'-'*50}")
    print(f"RELATÓRIO DE EXTRAÇÃO - {tipo_conteudo}:")
    print(f"  Pares processados: {pares_processados}")
    print(f"  Pares com erro: {pares_com_erro}")
    print(f"\nTaxa de Sucesso por Característica:")
    for carac, stats in sorted(stats_caracteristicas.items()):
        taxa = (stats['sucesso'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {carac:<20}: {stats['sucesso']}/{stats['total']} ({taxa:.1f}%)")
    print(f"{'-'*50}")

    return resultados

# --- 8. Função Principal ---

def criar_database_de_similaridade(diretorio_principal, arquivo_saida_csv):
    """Função principal melhorada com balanceamento e análise detalhada."""
    
    diretorio_principal = os.path.normpath(diretorio_principal)
    
    print("="*80)
    print("GERADOR AVANÇADO DE BANCO DE DADOS DE SIMILARIDADE MUSICAL")
    print("Versão com Múltiplas Características e Balanceamento Automático")
    print("="*80)
    print(f"Diretório base: {diretorio_principal}")
    
    # Define seed para reprodutibilidade
    random.seed(42)
    np.random.seed(42)
    
    # Processa diretórios
    resultados_plagios = processar_diretorio(diretorio_principal, "Plágios", eh_plagio=True)
    resultados_variadas = processar_diretorio(diretorio_principal, "Variadas", eh_plagio=False)
    
    # Combina resultados
    todos_resultados = resultados_plagios + resultados_variadas
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS ANTES DO BALANCEAMENTO:")
    print(f"  Total de pares de plágios: {len(resultados_plagios)}")
    print(f"  Total de pares variados: {len(resultados_variadas)}")
    print(f"  TOTAL GERAL: {len(todos_resultados)}")
    print(f"{'='*60}")

    if not todos_resultados:
        print("\nNenhum par processado. Encerrando.")
        return

    # Balanceamento automático
    todos_resultados = balancear_resultados(todos_resultados)
    
    # Análise de compatibilidade das características
    analisar_compatibilidade_caracteristicas(todos_resultados)

    # Salva os resultados
    df_resultados = pd.DataFrame(todos_resultados)

    try:
        df_resultados.to_csv(arquivo_saida_csv, index=False, encoding='utf-8')
        print(f"\n{'='*60}")
        print(f"BANCO DE DADOS SALVO COM SUCESSO!")
        print(f"Arquivo: {arquivo_saida_csv}")
        print(f"Total de registros: {len(df_resultados)}")
        print(f"{'='*60}")
        
        # Estatísticas finais detalhadas
        plagios_final = df_resultados[df_resultados['Plagio'] == 'sim']
        variadas_final = df_resultados[df_resultados['Plagio'] == 'não']
        
        print(f"\nDISTRIBUIÇÃO FINAL BALANCEADA:")
        print(f"  Plágios: {len(plagios_final)} amostras")
        print(f"  Variadas: {len(variadas_final)} amostras")
        print(f"  Balanceamento: {len(plagios_final)/len(df_resultados)*100:.1f}% vs {len(variadas_final)/len(df_resultados)*100:.1f}%")
        
        # Estatísticas por classe
        if len(plagios_final) > 0:
            sim_plagios = plagios_final['Similaridade'].astype(float)
            print(f"\nESTATÍSTICAS PLÁGIOS ({len(plagios_final)} amostras):")
            print(f"  Similaridade Média: {sim_plagios.mean():.4f} ({sim_plagios.mean()*100:.2f}%)")
            print(f"  Mediana: {sim_plagios.median():.4f} ({sim_plagios.median()*100:.2f}%)")
            print(f"  Desvio Padrão: {sim_plagios.std():.4f}")
            print(f"  Range: {sim_plagios.min():.4f} - {sim_plagios.max():.4f}")
        
        if len(variadas_final) > 0:
            sim_variadas = variadas_final['Similaridade'].astype(float)
            print(f"\nESTATÍSTICAS VARIADAS ({len(variadas_final)} amostras):")
            print(f"  Similaridade Média: {sim_variadas.mean():.4f} ({sim_variadas.mean()*100:.2f}%)")
            print(f"  Mediana: {sim_variadas.median():.4f} ({sim_variadas.median()*100:.2f}%)")
            print(f"  Desvio Padrão: {sim_variadas.std():.4f}")
            print(f"  Range: {sim_variadas.min():.4f} - {sim_variadas.max():.4f}")
        
        # Análise de separabilidade das classes
        if len(plagios_final) > 0 and len(variadas_final) > 0:
            sim_plagios = plagios_final['Similaridade'].astype(float)
            sim_variadas = variadas_final['Similaridade'].astype(float)
            
            diferenca_medias = abs(sim_plagios.mean() - sim_variadas.mean())
            std_pooled = np.sqrt((sim_plagios.var() + sim_variadas.var()) / 2)
            separabilidade = diferenca_medias / std_pooled if std_pooled > 0 else 0
            
            print(f"\nANÁLISE DE SEPARABILIDADE:")
            print(f"  Diferença entre médias: {diferenca_medias:.4f}")
            print(f"  Índice de separabilidade: {separabilidade:.3f}")
            if separabilidade > 1.0:
                print(f"  ✓ Boa separabilidade entre classes")
            elif separabilidade > 0.5:
                print(f"  ⚠ Separabilidade moderada")
            else:
                print(f"  ✗ Baixa separabilidade - classes sobrepostas")
        
        # Mostra amostra do dataset
        print(f"\n{'='*80}")
        print("AMOSTRA DO DATASET CRIADO:")
        print(f"{'='*80}")
        
        # Seleciona colunas principais para exibição
        colunas_principais = ['ID_do_Par', 'Musica_1', 'Musica_2', 'Similaridade', 'Plagio']
        colunas_caracteristicas = [col for col in df_resultados.columns if col.startswith('sim_')]
        
        # Mostra primeiras 5 linhas com colunas principais
        print("\nCOLUNAS PRINCIPAIS:")
        print(df_resultados[colunas_principais].head().to_string(index=False))
        
        # Mostra algumas características individuais
        if colunas_caracteristicas:
            print(f"\nCARACTERÍSTICAS INDIVIDUAIS (primeiras 5):")
            top_5_carac = colunas_caracteristicas[:5]
            print(df_resultados[['ID_do_Par'] + top_5_carac].head().to_string(index=False))
            
            if len(colunas_caracteristicas) > 5:
                print(f"\n... e mais {len(colunas_caracteristicas)-5} características disponíveis")
        
        # Recomendações de uso
        print(f"\n{'='*80}")
        print("RECOMENDAÇÕES DE USO:")
        print(f"{'='*80}")
        print("1. CARACTERÍSTICAS MAIS DISCRIMINATIVAS:")
        
        # Identifica as melhores características baseado na correlação
        melhores_caracteristicas = []
        for col in colunas_caracteristicas:
            try:
                valores = df_resultados[col].astype(float)
                labels = (df_resultados['Plagio'] == 'sim').astype(int)
                corr, p_val = pearsonr(valores, labels)
                if abs(corr) > 0.1 and p_val < 0.05:  # Correlação significativa
                    melhores_caracteristicas.append((col, abs(corr), p_val))
            except:
                continue
        
        melhores_caracteristicas.sort(key=lambda x: x[1], reverse=True)
        
        for i, (carac, corr, p_val) in enumerate(melhores_caracteristicas[:5]):
            carac_nome = carac.replace('sim_', '').replace('_', ' ').title()
            print(f"   {i+1}. {carac_nome}: correlação = {corr:.3f} (p = {p_val:.3f})")
        
        print(f"\n2. PARA MACHINE LEARNING:")
        print(f"   - Use as colunas 'sim_*' como features")
        print(f"   - Use a coluna 'Plagio' como target")
        print(f"   - Dataset já está balanceado")
        print(f"   - Total de features disponíveis: {len(colunas_caracteristicas)}")
        
        print(f"\n3. ANÁLISE ESTATÍSTICA:")
        print(f"   - Similaridade combinada na coluna 'Similaridade'")
        print(f"   - Características individuais nas colunas 'sim_*'")
        print(f"   - Metadados disponíveis: ID, Tipo_Diretorio, arquivos originais")
        
        # Salva também um arquivo de metadados sobre as características
        metadata_file = arquivo_saida_csv.replace('.csv', '_metadata.txt')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("METADADOS DO DATASET DE SIMILARIDADE MUSICAL\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Data de criação: {pd.Timestamp.now()}\n")
            f.write(f"Total de amostras: {len(df_resultados)}\n")
            f.write(f"Plágios: {len(plagios_final)}\n")
            f.write(f"Variadas: {len(variadas_final)}\n\n")
            
            f.write("CARACTERÍSTICAS EXTRAÍDAS:\n")
            f.write("-" * 30 + "\n")
            for nome, config in CARACTERISTICAS_CONFIG.items():
                if config['ativo']:
                    f.write(f"{nome}: {config['nome']}\n")
                    f.write(f"  Descrição: {config['descricao']}\n")
                    f.write(f"  Parâmetros: {config['parametros']}\n\n")
            
            if melhores_caracteristicas:
                f.write("MELHORES CARACTERÍSTICAS (por correlação):\n")
                f.write("-" * 40 + "\n")
                for i, (carac, corr, p_val) in enumerate(melhores_caracteristicas):
                    f.write(f"{i+1}. {carac}: correlação = {corr:.3f} (p = {p_val:.3f})\n")
        
        print(f"\nArquivo de metadados salvo: {metadata_file}")
        
    except Exception as e:
        print(f"\nERRO ao salvar arquivo CSV: {e}")

# --- 9. Função de Análise Rápida ---

def analisar_dataset_existente(arquivo_csv):
    """
    Função para analisar um dataset já criado sem reprocessar os áudios.
    """
    try:
        df = pd.read_csv(arquivo_csv)
        print(f"Dataset carregado: {len(df)} amostras")
        
        # Análise básica
        print(f"\nDistribuição de classes:")
        print(df['Plagio'].value_counts())
        
        # Características disponíveis
        colunas_sim = [col for col in df.columns if col.startswith('sim_')]
        print(f"\nCaracterísticas disponíveis: {len(colunas_sim)}")
        
        # Performance de cada característica
        print(f"\nPerformance das características:")
        for col in colunas_sim[:10]:  # Mostra top 10
            try:
                valores = df[col].astype(float)
                labels = (df['Plagio'] == 'sim').astype(int)
                corr, p_val = pearsonr(valores, labels)
                print(f"{col.replace('sim_', ''):<20}: {corr:>6.3f} (p={p_val:.3f})")
            except:
                continue
                
    except Exception as e:
        print(f"Erro ao analisar dataset: {e}")

# --- 10. Execução Principal ---
if __name__ == "__main__":
    # --- CONFIGURAÇÃO ---
    DIRETORIO_RAIZ = 'Plágios-2'
    ARQUIVO_CSV_DE_SAIDA = 'database_similaridade_musical_avancado.csv'
    
    # Opção para apenas analisar um dataset existente
    APENAS_ANALISAR = False  # Mude para True para analisar dataset existente
    
    if APENAS_ANALISAR and os.path.exists(ARQUIVO_CSV_DE_SAIDA):
        print("Modo de análise: analisando dataset existente...")
        analisar_dataset_existente(ARQUIVO_CSV_DE_SAIDA)
    else:
        print("Modo de criação: processando áudios e criando dataset...")
        criar_database_de_similaridade(DIRETORIO_RAIZ, ARQUIVO_CSV_DE_SAIDA)