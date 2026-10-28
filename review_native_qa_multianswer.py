"""Explicit assistant-authored support decisions for the unchanged 192 QAs.

Anchors below were selected after reading every train/development passage and
question. They are not selected by model scores or by an entailment heuristic.
Sentence expansion computes offsets only. Additional references are enumerated
explicitly, never inferred from nearby names or numbers.
"""
from native_qa_multianswer import *


def specifications():
    decisions = {'train': {}, 'development': {}}
    def add(split, index, *anchors, alternatives=(), include=None, note=None):
        if index in decisions[split]:
            raise ValueError('Duplicate manual review')
        decisions[split][index] = dict(anchors=list(anchors), alternatives=list(alternatives),
                                      include=include or {}, note=note)
    def t(index, *anchors, **kwargs):add('train',index,*anchors,**kwargs)
    def d(index, *anchors, **kwargs):add('development',index,*anchors,**kwargs)

    t(0,'reinforcing the phonetic transparency of Spanish')
    t(1,'in "Lesson 2," students hear the sound of a')
    t(2,'were established by the General Assembly in 1993-94')
    t(3,'information was collected and analyzed from the 30 projects')
    t(4,'Several other states, including Ohio, California, and Florida')
    t(5,'Through the end of July, Michigan has seen 546 cases')
    t(6,'In 2018, current use of any tobacco product')
    t(7,'In 2018, current use of any tobacco product','During 2017–2018, current use of any tobacco product')
    t(8,'set new GHG emissions and sustainability goals for 2030')
    t(9,'In 2023, funding was expanded to implement')
    t(10,"along the Route's 2,448 miles")
    t(11,'Museum was dedicated on July 4, 2000')
    t(12,'Poisson-based regression models (including quasi-Poisson and negative binomial)')
    t(13,'Ariel, Barak, William A. Farrar, and Alex Sutherland.',note='Citation author inversion and punctuation are preserved exactly; the first citation is the requested study.')
    t(14,'Index testing, also known as partner notification or contact tracing')
    t(15,'leaving 9.4 million persons with undiagnosed infection')
    t(16,'Nature Sketching and Nature Journals:',('The workshop will be led by art instructor Georgia Braun.',0),
      note='The requested workshop lead is explicit. The question calls the person a naturalist; the passage calls her an art instructor. The answer identifies the workshop lead, without asserting that the profession was independently verified.')
    t(17,'Acrylic or Watercolor Workshops for those 16 years and older')
    t(18,'in 1789 it was reconstructed and became the seat')
    t(19,'George Washington take the oath of office as our first President')
    t(20,'was awarded $91,000 by the US Environmental Protection Agency','The Vermont Energy Education Program in Montpelier received the funding')
    t(21,'was awarded $91,000 by the US Environmental Protection Agency','The Vermont Energy Education Program in Montpelier received the funding')
    t(22,'A photograph of artist, Charles Mazoujian, sketching an F-105')
    t(23,'Former Senator Edward Kennedy (1-68-577)')
    t(24,'smoking among Michigan teens has dropped by 28 percent','Teen smoking rates have not only dropped 28 percent since 1997')
    t(25,'27.6 percent of 9-12 graders smoked in 2001')
    t(26,'from the 16 cases reported from all 50 states during 2013-2015')
    t(27,'were hospitalized for a median of 3 days')
    t(28,'delegated by the HHS secretary to the Administration for Children and Families (ACF)','ACF serves as the lead agency during both emergency and non-emergency activities')
    t(29,'was established in 1935 under Section 1113')
    t(30,'Since 1972, the national Senior Nutrition Program has supported nutrition services',note='The passage dates the program support to 1972; no additional founding history is inferred.')
    t(31,'Funded by the Older Americans Act')
    t(32,'J. R. Climer of Madison County, Mississippi, became the first resident')
    t(33,'Three weeks before Christmas of 1903')
    t(34,'Date Published','November 2011')
    t(35,'Length','17 pages')
    t(36,'on property owned by Purdue University')
    t(37,'Trees were planted 8 feet apart in alternating rows',alternatives=('8 feet','8 feet apart'))
    t(38,'Among the 474 individuals who had been offered vaccination, 397 (83.8%) received')
    t(39,'having heard positive information about the vaccine (RR 1.30, 95% CI 1.06-1.60)')
    t(40,'collaborates with Gallaudet University and AppTek')
    t(41,'a company formed in September 2022')
    t(42,'This long, narrow bay system stretches 130 miles')
    t(43,'Episode 26, Season 39 of Texas Parks & Wildlife','On this special episode of Texas Parks and Wildlife',alternatives=('Texas Parks and Wildlife',))
    t(44,'The Vermont Agency of Agriculture, Food and Markets (VAAFM) announces')
    t(45,'The Farm to School and Early Childhood Vision Grant supports projects')
    t(46,'the Pacific Decadal Oscillation is a much larger symphony')
    t(47,'approximately every 15 to 20 years',note='The excerpt supports the reported period, and names Chao. His full name and the shorter-cycle modifier appear in the question only; no extra cycle is inferred.')
    t(48,'Festival Beach Food Forest is a community garden in downtown Austin')
    t(49,'Every second Saturday of the month, volunteers practice permaculture')
    t(50,'General backcountry visitation is allowed','length of stay up to 14 days')
    t(51,'Stock use is allowed with a maximum party size of 20 and 20 head of stock.',alternatives=('20 head of stock',),note='Only the stock count is accepted; the separate party-size count is not evidence for head of stock.')
    t(52,'Trot along with a young Coyote as he lopes')
    t(53,'Learn about Navajo Culture from Morning Star, the Navajo Shepherd Girl')
    t(54,'Shaded Relief with Height as Color, Mount Meru, Tanzania')
    t(55,'It reaches 4,566 meters (14,978 feet) in height')
    t(56,'Amboy Crater was designated a National Natural Landmark in 1973')
    t(57,'The Amboy Crater, formed of ash and cinders, is 250 feet high')
    t(58,'filling up my truck in Alaska for $109 bucks',note='The quoted truck amount is explicit. The question supplies the speaker name; the excerpt itself does not establish that attribution.')
    t(59,'Had it been fully empty, it would have been $142 dollars',note='The hypothetical full-tank amount is explicit. The speaker attribution is question-provided.')
    t(60,'providing physical care (54 minutes)')
    t(61,'In 2023, adults living in households with children under age 6')
    t(62,'Between 1970-1982, total catch','34 percent carp')
    t(63,'until by 1983 over 2,250 Iowans were licensed')
    t(64,'In 2005, an act of nature breached a dike')
    t(65,'The Spencer Island Unit is located in the Snohomish River estuary just east of Everett.',
      alternatives=('the Snohomish River estuary just east of Everett','Snohomish River estuary just east of Everett'))
    t(66,'Enceladus Flyby at a Glance','Altitude 1,142 miles (1,839 km)',note='The altitude belongs to the displayed Enceladus flyby. The E-20 identifier is question-provided, not independently present in the excerpt.')
    t(67,'Enceladus Flyby at a Glance','Speed (rel. to Enceladus) 19,014 mph (8.5 km/sec)',note='The speed belongs to the displayed flyby; the E-20 identifier is question-provided.')
    t(68,'Overall, 24 states had unemployment rates lower')
    t(69,'Nevada had the highest unemployment rate')
    t(70,'Common catsear is originally from Europe')
    t(71,'Common catsear can produce as many as 6,000 seeds per rosette')
    t(72,'Gay Marine Corps Veteran Dave Mapp','Marine Corps Veteran Dave Mapp, say the care')
    t(73,'Mapp was a guest during Bay Pines VA’s Pride Month observance',
      'LGBTQ+ Veterans and their dependents are welcome and fully supported by the Bay Pines VA.',
      note='The event paragraph identifies the facility in both the guest description and the guest’s quotation. Both refer to the same Bay Pines VA. The possessive VA’s mention is retained as supporting evidence even if native word boundaries prevent copying its shorter name there.')
    t(74,'Black Crappie','Under 9”',('8',0),note='The selected contiguous table block associates Black Crappie, Under 9 inches and 8 monthly servings; other fish rows are excluded.')
    t(75,'Muskellunge','Do Not Eat',note='The contiguous Muskellunge/Mercury/Any/Do Not Eat table block provides the relation.')
    t(76,'From September to December, vaccination intent (defined as being absolutely certain or very likely')
    t(77,'As of February 8, 2021, 59.3 million doses')
    t(78,'Low End Amount – 9 in 10 Chance (90%)','or 90% chance that more snow will fall')
    t(79,'only a 1 in 10, or 10% chance that less snow will fall',include={'10%':[0]})
    t(80,'The USGS National Geospatial Program (NGP) provides vital')
    t(81,'The National Map, a suite of geospatial products describing the Nation’s topography',include={'The National Map':[2]})
    t(82,'The task force is chaired by Thomas Stallworth')
    t(83,'co-chaired by David Sanders, executive vice president of Systems Improvement at Casey Family Programs')
    t(84,'For the past 10 years Thomas Anderson (Trey) has been a guest speaker')
    t(85,'View policy open for public comment on regulations.gov')
    t(86,'Central Utah’s health district has a lower rate at 6.8%')
    t(87,'8% of the adult population have diagnosed diabetes')
    t(88,'Global Hepatitis Outbreak and Surveillance Technology (GHOST)')
    t(89,'derived from the HCV E1/E2-junction genomic region')
    t(90,'a total of 2239 new HPO terms')
    t(91,'The Medical Action Ontology (MAxO) is a new effort')
    t(92,'Colorectal cancer is one of the most common cancers')
    t(93,"If you're 45 or older")
    t(94,'these projects will serve 199 school districts and educational entities')
    t(95,'will serve 1,650,985 students')
    t(96,'county—Wayne')
    t(97,'There are two optional unified\ncounties—Bay and Oakland')
    t(98,'95% of surface water in the U.S.')
    t(99,'35 million people in the Great Lakes Basin rely')
    t(100,'The earliest isolates found in 1983 and 1984 belong to South Pacific genotype III')
    t(101,'More recent isolates obtained during surveillance efforts during 1997–2004 belong to Africa/America genotype IV',note='More recent isolates is the source’s explicit label for 1997–2004, not a claim about the later 2009 introduction.')
    t(102,'The Bureau of Indian Affairs (BIA), established in 1824')
    t(103,'Bonneville Power Administration','A self-funded federal agency that markets power from hydroelectric projects')
    t(104,'two Forest Service Regions (Northern and Intermountain)')
    t(105,'now encompasses a total of 2,366,757 acres')
    t(106,'Sgt. 1st Class Alwyn C. Cashe','the first Black recipient of the medal since 9/11',note='The final he refers back to the named Cashe; the evidence includes both antecedent and award claim.')
    t(107,'Cashe was posthumously awarded the Medal of Honor in December 2021')
    t(108,'where ECOSTRESS recorded some of the hottest land surface temperatures',note='The recorder is explicitly ECOSTRESS; NASA ownership is supplied by the question and not independently established in this excerpt.')
    t(109,'At the lower right of the image is Phoenix Sky Harbor International Airport')
    t(110,'The tool was initially licensed by ZEISS')
    t(111,'Among the most promising fuels for new advanced reactor designs is Tri-structural ISOtropic particle fuel, or TRISO.',alternatives=('Tri-structural ISOtropic particle fuel','TRISO'))
    t(112,'Adults aged 18-24 years were least likely to report having received a COVID-19 vaccine')
    t(113,'Concerns about vaccine safety and effectiveness were the primary reported reasons')
    t(114,'The Michigan Tobacco Quitline began operations in 2003')
    t(115,'Tobacco use is the leading cause of preventable illness in our state')
    t(116,'called Clavusodens mcginnisi')
    t(117,'fossil teeth collected from deep inside the Ste. Genevieve Formation')
    t(118,'The Headwaters Forest Reserve protects 7,472 acres')
    t(119,'Established on March 1, 1999, the Reserve is managed')
    t(120,'Postcard From Texas: Passing Time at Ray Roberts Lake State Park','Stories from this episode:',note='The excerpt explicitly names the park and calls the content this episode. Episode 22/Season 40 is a question-provided identifier, not a verified statement of this fixed passage.')
    t(121,'park north of Dallas and Fort Worth',note='The cities are explicit. Surprisingly close and the feeling of remoteness are framing supplied by the question, not a measured distance claim.')
    t(122,'Moderna’s vaccine for children under six calls for 25 micrograms per shot')
    t(123,'Moderna’s vaccine requires two doses','The second shot should be administered 28 days after the first dose')
    t(124,'This year\'s "Ultimate Ascent" challenge',note='The challenge is explicitly named. The year 2013 is question-provided and absent from this excerpt.')
    t(125,'Beverly Hills High School in Beverly Hills, Calif., received the competition\'s highest award')
    t(126,'86 organisms from the phylum Chloroflexi that represent 15 distinct lineages')
    t(127,'86 organisms from the phylum Chloroflexi','Chloroflexi, a relatively unknown phylum')

    d(0,'The U.S. Forest Service is working with The Wildlife Society')
    d(1,'Eastern Forest Environmental Threat Assessment Center')
    d(2,'Her grandmother was related to Jonathon Walker','He was captured and branded for helping slaves escape')
    d(3,'Ernie Cole, the blacksmith in Harwich Port')
    d(4,'enumerate the 2018-19 spawning run of Steelhead, Chinook Salmon, and Coho Salmon')
    d(5,'Population Estimates 2018-19','Chinook Salmon - 3,831')
    d(6,'runs from 8:00am-3:00pm, daily')
    d(7,'with a minimum of 12 students for the class to run')
    d(8,'With the heat index expected to be at 104 degrees tomorrow and Sunday, July 18 and 19')
    d(9,'Baltimore City Health Commissioner Dr. Leana Wen has issued a Code Red Heat Advisory')
    d(10,'So far this year they have completed 889 lessons')
    d(11,'Currently, 254 students use our computer science software')
    d(12,"one of Louis Peele's gunhawks","Meanwhile, Peele raided the Circle Crossbar")
    d(13,'out of Great Lost Valley rode Gage Gardin','Gage hit the backtrail with flaring six-shooters')
    d(14,'Designated as a Historic District by the El Paso City Council in 1985')
    d(15,'Spanning 65 acres')
    d(16,'boundary between Texas and Oklahoma in 1921')
    d(17,'In 1996, Texas and Oklahoma agreed\nto create the Red River Boundary Commission')
    d(18,'WHEN: May & June 2023—all sessions take place on Wednesdays at 1 p.m. ET')
    d(19,'WHEN: May & June 2023—all sessions take place on Wednesdays at 1 p.m. ET')
    d(20,'particularly when working with Minnesota’s three largest ethnic minority people groups: Latino, Hmong, and Somali.',
      alternatives=('Latino','Hmong'),note='The question asks for one group. Each listed group is independently a complete valid answer; no exclusive preference for Somali is inferred.')
    d(21,'Who is this for?','Healthcare providers and professionals working with diverse populations who have experienced trauma.',
      alternatives=('Healthcare providers and professionals working with diverse populations who have experienced trauma',))
    d(22,'persons triply diagnosed with HIV, mental illness and substance use')
    d(23,'Overall, we found a relative 13% improvement in both retention')
    d(24,'Between 2004 and 2015, liver cancer incidence increased 46 percent')
    d(25,'Chronic hepatitis B and hepatitis C are leading causes of liver cancer')
    d(26,'One of the most courageous things JFK did was running for president.',alternatives=('running for president','running for president.'))
    d(27,'He was facing the favorite republic and current senator: Henry Cabot Lodge Jr.')
    d(28,'George Washington and Benjamin Franklin were the unlikeliest of rebels')
    d(29,'George Washington and Benjamin Franklin were the unlikeliest of rebels','Washington in the 1770s stood at the apex','Franklin was more successful still')
    d(30,'sales tax funds open space, with revenue shared between the County (50%) and its eight municipalities (50%)',include={'50%':[0]},note='Only the County allocation occurrence is labeled. The identical municipality percentage is a different recipient.')
    d(31,'The second-largest revenue source for Larimer County is sales tax','Sales tax revenue is projected',
      alternatives=('sales tax','Sales Tax'),note='The paragraph uses three capitalization surfaces for the same revenue source; each exact source surface is accepted explicitly, without global case folding.')
    d(32,'The Eagle commenced weekly publication in Silver City, New Mexico, on August 22, 1894')
    d(33,'Silver City had been established in 1870 by a Captain Hurlburt')
    d(34,'Rangers Corey and Curcio located the subject',alternatives=('Corey','Curcio','Corey and Curcio'),
      note='The singular Which Ranger question does not distinguish the two named responders. Either named responder or their joint response is accepted.')
    d(35,'Town of Fenton Broome County Motor Vehicle Accident/Medical Event:','Ranger Oldroyd began administering CPR')
    d(36,'According to the 2000 census, Michigan has nearly 1.6 million people over the age of 60')
    d(37,'celebrate November as Family Caregiver Month')
    d(38,'Eastern United States wildfire hazard model: 2000-2009','during the period 2000 to 2009',alternatives=('2000 to 2009',),note='Both the heading interval and the same interval written out in the parameterization sentence are accepted.')
    d(39,'the region of New Jersey, Ohio, and Pennsylvania')
    d(40,'a Consent Decree that was entered with the U.S. District Court','Eastern Division in July 2012')
    d(41,'This Consent Decree, valued at $5.5 million')
    d(42,'it’s launched 154 times')
    d(43,'The ICESat-2 spacecraft and ELaNa XVIII payload will be carried into space')
    d(44,'Meet the Working Dogs for Conservation, a canine team','Working Dogs for Conservation - 1:20','Working Dogs for Conservation are sniffing boats',note='Both descriptions and the intervening segment heading refer to the same named canine team.')
    d(45,'Zebra Mussels were first detected in Texas in 2009')
    d(46,'AC 120-125 is the advisory circular')
    d(47,'The MEL is directed by 14 CFR Part 91K, 121, 125 and 135')
    d(48,'Among CMs, getting the Ebola vaccine was associated','RR 1.43, 95% CI 1.05-1.94')
    d(49,'Most HCWs (81.7%) and 53.6% of CMs felt at risk')
    d(50,'Bullseye: Archery in Schools - 6:32','Follow a team of student archers from San Antonio',note='The target is the titled segment that follows the team; National Archery in Schools Program is a different named organization in the description.')
    d(51,'Browns Creek: Postcard From Texas - 24:34')
    d(52,'The Boeing 767 included a low-fuel system warning at 2,000 pounds total fuel')
    d(53,'The Boeing 767 included a low-fuel system warning')
    d(54,'The site of an April 4, 2010, magnitude 7.2 earthquake','site of a 7.2 quake on April 4','On April 4, 2010, a magnitude 7.2 earthquake struck')
    d(55,'On April 4, 2010, a magnitude 7.2 earthquake struck along this fault about 64 kilometers (40 miles) south',note='The supplied distance is the earthquake location along the named fault, not a distance asserted for every point along the whole fault.')
    d(56,'1999’s score of 5.19 to 2002’s 5.75')
    d(57,'80 percent of the preschool classrooms had global quality scores of 5 or higher')
    d(58,'providing students and families with basic needs','in 20,654 cases',note='The program/count is explicit; the 2016-17 reporting-year identifier is question-provided.')
    d(59,'Pathways to Potential had 131,307 interventions with 45,975 individuals',note='The count is explicit; the 2016-17 reporting year is question-provided.')
    d(60,'Properties owned by public agencies or designated as agricultural land are exempt')
    d(61,'Measure Q is funded by an annual $87 per parcel tax')
    d(62,'the researchers used Landsat data','they estimated forest age and disturbance over decades')
    d(63,'analyze tree cover annually from 1984-2020')
    return decisions


def materialize(output=None):
    verify_history()
    destination = Path(output) if output is not None else OUT
    specs = specifications()
    decisions = {}
    for split, count in [('train',128),('development',64)]:
        source = rows(split)
        if set(specs[split]) != set(range(count)):
            raise ValueError('Every training/development question needs an explicit manual decision')
        for index,row in enumerate(source):
            spec = specs[split][index]
            support = sorted({tuple(anchor_support(row['context'],a)) for a in spec['anchors']})
            # A list of explicit anchors forms a supporting block, including
            # intervening table cells and heading-to-description links.
            support = [[min(a for a,b in support),max(b for a,b in support)]]
            options = []
            for text in [row['answer']] + spec['alternatives']:
                occurrences = intervals(row['context'],text)
                if text in spec['include']:
                    occurrences = [occurrences[i] for i in spec['include'][text]]
                occurrences = [[a,b] for a,b in occurrences if any(c<=a<b<=d for c,d in support)]
                options.append(dict(text=text,occurrences=occurrences,
                    reason='Original reference, checked against the selected evidence.' if text==row['answer'] else
                    'Explicitly reviewed alternative supported by the same requested fact; see the question-specific review note.'))
            entry = dict(fingerprint=qa_fingerprint(row),split=split,index=index,source_group=row['source_group'],
                original_answer=row['answer'],decision='retain',support=support,answers=options,
                reason='Manually reviewed question-to-evidence link: '+'; '.join(a[0] if isinstance(a,(list,tuple)) else a for a in spec['anchors']),
                note=spec['note'],review_anchors=spec['anchors'])
            try:
                validate_entry(row,entry)
            except (ValueError,IndexError) as error:
                raise ValueError(f'{split}/{index}: {error}') from error
            decisions[entry['fingerprint']] = entry
    overlay = dict(version=4,task='passage_grounded_extractive_multianswer_qa',
        source_manifest_sha256=sha(OLD/'data/frozen_manifest.json'),
        supersedes='V3 single-answer support stop remains preserved; this is a newly authorized contract.',
        reviewer='assistant; no independent human review claimed',model_predictions_used=False,
        locked_test_questions_reviewed=0,original_questions_answers_and_splits_unchanged=True,
        acceptance_scope='Exact case/punctuation-sensitive match to explicitly approved contiguous surfaces; not exhaustive semantic paraphrase scoring.',
        question_referent_policy='Uncontradicted question-provided identifiers may resolve the supplied excerpt; only the requested answer relation is credited. Unverified qualifiers are disclosed per record, not claimed as source-verified facts.',
        decisions=decisions)
    overlay['review_revision'] = dict(previous_draft='support_overlay.json',
        previous_draft_sha256=sha(OUT/'support_overlay.json'),
        changes='Train 73 evidence includes the guest’s same-event quotation identifying the same facility. Supported but uncopyable mentions remain in the ledger; no reference is removed for compatibility.',
        trigger='Character/native-boundary diagnostic, not model selection or predictions')
    if destination.resolve()!=OUT.resolve():
        draft = destination/'support_overlay.json'
        draft.parent.mkdir(parents=True,exist_ok=True)
        with draft.open('xb') as stream:
            stream.write((OUT/'support_overlay.json').read_bytes())
    write_new(destination/'reviewed_answers.json',overlay)
    verify_overlay(destination/'reviewed_answers.json')
    return overlay


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    overlay=materialize(args.output)
    print('REVIEW_OVERLAY_COMPLETE',len(overlay['decisions']))
