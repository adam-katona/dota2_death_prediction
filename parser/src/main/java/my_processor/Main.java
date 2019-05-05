package my_processor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.qos.logback.core.joran.conditional.ElseAction;

import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.List;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import skadistats.clarity.model.Entity;
import skadistats.clarity.model.FieldPath;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.entities.OnEntityPropertyChanged;
import skadistats.clarity.processor.entities.OnEntityCreated;
import skadistats.clarity.processor.entities.OnEntityUpdated;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.source.MappedFileSource;
import skadistats.clarity.processor.reader.OnMessage;
import skadistats.clarity.processor.reader.OnTickEnd;
import skadistats.clarity.processor.reader.OnTickStart;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.decoder.Util;

import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.model.StringTable;
import skadistats.clarity.processor.stringtables.StringTables;
import skadistats.clarity.processor.stringtables.UsesStringTable;

import java.util.Optional;

// Overview
// The purpuse of this program is to take a replay file, and save a timeseries of a set of attributes.
// We need 2 things for this.
// - A list of attributes for each kind of game object: these are hardcoded in constant list. (these could be an input if it is needed)
// - And the hierarchy of game object we are interested in: (for example a hero have abilities...) This is hardcoded as well.

// The life state variable is recorded with full resolution, while the other attributes are sample (by default take every 4th.)

// WARNING: The parser is quite fragile, it relies on every attribute being present on the first try (after all the checks).
// If this is not the case, the names and the values do not line up... At least it will complain about the error :)

@UsesEntities
public class Main {

    Boolean error_has_occured = false;
    String last_error = "";
    Set<String> error_set = new HashSet<String>();
    Set<String> item_names_set = new HashSet<String>();

    int numPlayers = 10;
    int[] validIndices = new int[numPlayers];

    boolean init = false; // are all the entities present already?

    Boolean feature_list_done = false; 
    List<String> feature_names = new ArrayList<String>();

    FileWriter cvs_file;
    FileWriter life_state_cvs_file;

    final Boolean save_full_res_lifestate = true; 

    final int TICKS_PER_DATAPOINTS = 4; // change this to modify the sampling rate
    final int NUM_ABILITIES = 8;  // read the first 8 (out of 28)
    final int NUM_ITEMS = 13; // is it enough to check only the first few???
    final String[] player_resources_attributes = 
    {
        "m_vecPlayerTeamData.%i.m_nSelectedHeroID",
        //"m_vecPlayerTeamData.%i.m_bHasRandomed", dont care
        //"m_vecPlayerTeamData.%i.m_bHasPredictedVictory", always zero
        "m_vecPlayerTeamData.%i.m_iFirstBloodClaimed",
        "m_vecPlayerTeamData.%i.m_flTeamFightParticipation",
        "m_vecPlayerTeamData.%i.m_iLevel",
        "m_vecPlayerTeamData.%i.m_iKills",
        "m_vecPlayerTeamData.%i.m_iDeaths",
        "m_vecPlayerTeamData.%i.m_iAssists"
    };

    final String[] team_attributes = 
    {
        "m_vecDataTeam.%i.m_iObserverWardsPlaced",
        "m_vecDataTeam.%i.m_iSentryWardsPlaced",
        "m_vecDataTeam.%i.m_iCreepsStacked",
        "m_vecDataTeam.%i.m_iCampsStacked",
        "m_vecDataTeam.%i.m_iRunePickups",
        "m_vecDataTeam.%i.m_iTowerKills",
        "m_vecDataTeam.%i.m_iRoshanKills",
        "m_vecDataTeam.%i.m_iTotalEarnedGold",
        "m_vecDataTeam.%i.m_iLastHitCount",
        "m_vecDataTeam.%i.m_iTotalEarnedXP",
        "m_vecDataTeam.%i.m_fStuns"
    };

    final String[] hero_attributes = 
    {

        "CBodyComponent.m_cellX",  // int
        "CBodyComponent.m_vecX",   // float
        "CBodyComponent.m_cellY",  // int
        "CBodyComponent.m_vecY",   // float


        "m_flAgility",
        "m_flAgilityTotal",
        //"m_flCapsuleRadius",
        //"m_flFieldOfView",
        "m_flIntellect",
        "m_flIntellectTotal",
        "m_flMagicalResistanceValue", 
        "m_flMana",
        "m_flMaxMana",
        //"m_flNextAttack",   null?
        "m_flPhysicalArmorValue",   
        "m_flStrength",
        "m_flStrengthTotal",
        "m_flTauntCooldown",
        //"m_flRevealRadius",   null?

        "m_iBKBChargesUsed",
        "m_iAbilityPoints",
        "m_iPrimaryAttribute",
        "m_iMoveSpeed",
        "m_iMaxHealth",
        "m_iHealth",
        "m_iDamageBonus",
        "m_iDamageMax",
        "m_iDamageMin",

        "m_lifeState", // int

        "m_iTaggedAsVisibleByTeam"
    };

    final String[] hero_ability_attributes = 
    {
        "m_iLevel",
        "m_iCastRange",
        "m_iManaCost",

        "m_fCooldown",
        //"m_fCooldownLength",  null?

        "m_bActivated",
        "m_bToggleState"
    };

    final String[] tower_attributes = 
    {
        "CBodyComponent.m_cellX",  // int
        "CBodyComponent.m_vecX",   // float
        "CBodyComponent.m_cellY",  // int
        "CBodyComponent.m_vecY",   // float

        "m_iHealth",
        "m_lifeState" // int

        // team num will be in the tower name
    };



    final String[] op_activatable_items = 
    {
        "item_blink",
        "item_black_king_bar",
        "item_magic_wand",
        "item_quelling_blade",
        "item_power_treads",
        "item_hand_of_midas",
        "item_hurricane_pike",
        "item_force_staff",
        "item_abyssal_blade",
        "item_mask_of_madness",
        "item_nullifier",
        "item_travel_boots",
        "item_dagon_5",
        "item_lotus_orb"
        // consumables
    };
    final String[] op_consumable_items = 
    {
        "item_tpscroll",
        "item_smoke_of_deceit",
        "item_clarity"
    };


    
    // normally we get the object from a few mater (manager) object, but the tower object are collected via callbacks. 
    // bacause i did not found them in the manager object.
    private SortedMap<String,Entity> latest_tower_entities = new TreeMap<String,Entity>();

    private boolean isTower(Entity e) 
    {
        return e.getDtClass().getDtName().startsWith("CDOTA_BaseNPC_Tower");

    }

    @OnEntityCreated
    public void onCreated(Entity e) 
    {
        if (!isTower(e)) 
        {
            return;
        }
        FieldPath fp_x = e.getDtClass().getFieldPathForName("CBodyComponent.m_cellX");
        Integer pos_x = e.getPropertyForFieldPath(fp_x);
        FieldPath fp_y = e.getDtClass().getFieldPathForName("CBodyComponent.m_cellY");
        Integer pos_y = e.getPropertyForFieldPath(fp_y);
        FieldPath fp_tower_team = e.getDtClass().getFieldPathForName("m_iTeamNum");
        Integer tower_team = e.getPropertyForFieldPath(fp_tower_team);

        // the coordinates are used as a unique name
        String tower_name = "Tower_" + tower_team.toString() + "_" + pos_x.toString() + "_" + pos_y.toString();
        
        latest_tower_entities.put(tower_name, e);
    }


    @OnEntityUpdated
    public void onUpdated(Entity e, FieldPath[] updatedPaths, int updateCount) 
    {
        if (!isTower(e)) 
        {
            return;
        }
        FieldPath fp_x = e.getDtClass().getFieldPathForName("CBodyComponent.m_cellX");
        Integer pos_x = e.getPropertyForFieldPath(fp_x);
        FieldPath fp_y = e.getDtClass().getFieldPathForName("CBodyComponent.m_cellY");
        Integer pos_y = e.getPropertyForFieldPath(fp_y);
        FieldPath fp_tower_team = e.getDtClass().getFieldPathForName("m_iTeamNum");
        Integer tower_team = e.getPropertyForFieldPath(fp_tower_team);

        String tower_name = "Tower_" + tower_team.toString() + "_" + pos_x.toString() + "_" + pos_y.toString();
        
        latest_tower_entities.put(tower_name, e);
    }


    public void try_init(Entity pr)
    {
        // NOTE: I am not sure what was going on here, my replays does not seem to have this issue
        // I kept the code nonetheless
        if (pr != null) 
        {
            //Radiant coach shows up in vecPlayerTeamData as position 5
            //all the remaining dire entities are offset by 1 and so we miss reading the last one and don't get data for the first dire player
            //coaches appear to be on team 1, radiant is 2 and dire is 3?
            //construct an array of valid indices to get vecPlayerTeamData from
            if (!init) 
            {
                int added = 0;
                int i = 0;
                //according to @Decoud Valve seems to have fixed this issue and players should be in first 10 slots again
                //sanity check of i to prevent infinite loop when <10 players?
                while (added < numPlayers && i < 100) {
                    try 
                    {
                        //check each m_vecPlayerData to ensure the player's team is radiant or dire
                        int playerTeam = getEntityProperty(pr, "m_vecPlayerData.%i.m_iPlayerTeam", i);
                        int teamSlot = getEntityProperty(pr, "m_vecPlayerTeamData.%i.m_iTeamSlot", i);
                        Long steamid = getEntityProperty(pr, "m_vecPlayerData.%i.m_iPlayerSteamID", i);
                        //System.err.format("%s %s %s: %s\n", i, playerTeam, teamSlot, steamid);
                        if (playerTeam == 2 || playerTeam == 3) {
                            //add it to validIndices, add 1 to added
                            validIndices[added] = i;
                            added += 1;
                        }
                    }
                    catch(Exception e) 
                    {
                        //swallow the exception when an unexpected number of players (!=10)
                        //System.err.println(e);
                    }

                    i += 1;
                }
                init = true;
            }
        }
    }

    void addSingleFeature(List<Float> feature_vec,float value,String feature_name)
    {
        saveFeatureName(feature_name,null);
        feature_vec.add(value);
    }
    void addMultipleFeature(List<Float> feature_vec,List<Float> values,String feature_name)
    {
        if(this.feature_list_done == false) // also do a check outside of the loop for performace
        {
            for(Integer i = 0 ; i < values.size(); i++)
            {
                saveFeatureName(feature_name + "_" + i.toString(),null);
            }
        }
        feature_vec.addAll(values);
    }
    
    public Float[] get_hero_positions(Entity hero)
    {
        Integer cellX = hero.getProperty("CBodyComponent.m_cellX");
        Float vecX = hero.getProperty("CBodyComponent.m_vecX");
        Float worldX = (cellX * 128.0f) + vecX;
        Float mapX = worldX - 16384.0f;
        Integer cellY = hero.getProperty("CBodyComponent.m_cellY");
        Float vecY = hero.getProperty("CBodyComponent.m_vecY");
        Float worldY = (cellY * 128.0f) + vecY;
        Float mapY = worldY - 16384.0f;

        Float[] pos = {mapX,mapY};
        return pos;
    }



    public void addEntityPropertyToFeatureVec(List<Float> data_point,Entity e,String attribute,Integer index,String name_prefix)
    {
        if(attribute.contains("m_i") || attribute.contains("m_n"))
        {
            addIntEntityPropertyToFeatureVec(data_point,e,attribute,index,name_prefix);
        }
        else if(attribute.contains("m_f"))
        {
            addFloatEntityPropertyToFeatureVec(data_point,e,attribute,index,name_prefix);
        }
        else if(attribute.contains("m_b"))
        {
            addBoolEntityPropertyToFeatureVec(data_point,e,attribute,index,name_prefix);
        }
        else if(attribute.contains("CBodyComponent"))
        {
            if (attribute.contains("cell"))
            {
                addIntEntityPropertyToFeatureVec(data_point,e,attribute,index,name_prefix);
            }
            else if(attribute.contains("vec"))
            {
                addFloatEntityPropertyToFeatureVec(data_point,e,attribute,index,name_prefix);
            }
            else { System.out.println("ERROR unknown CBodyComponent attribute: " + attribute);}
        }
        else if(attribute.contains("m_lifeState"))
        {
            addIntEntityPropertyToFeatureVec(data_point,e,attribute,index,name_prefix);
        }
        else
        {
            System.out.println("ERROR unknown type: " + attribute);
        } 
    }

    @UsesStringTable("EntityNames")
    @UsesEntities
    @OnTickStart
    public void onTickStart(Context ctx, boolean synthetic) 
    {
            
        Entity grp = ctx.getProcessor(Entities.class).getByDtName("CDOTAGamerulesProxy");
        Entity pr = ctx.getProcessor(Entities.class).getByDtName("CDOTA_PlayerResource");
        Entity dData = ctx.getProcessor(Entities.class).getByDtName("CDOTA_DataDire");
        Entity rData = ctx.getProcessor(Entities.class).getByDtName("CDOTA_DataRadiant");

        try_init(pr);
        int tick = ctx.getTick();
        Float time = 0.0f;
        if (grp != null) 
        {
            time = (float) getEntityProperty(grp, "m_pGameRules.m_fGameTime", null);
        }



        if(init == true) 
        {
            if (save_full_res_lifestate == false && (tick % this.TICKS_PER_DATAPOINTS) != 0)
            {
                return;
            }

            List<Float> lifeStateData = new ArrayList<Float>();
            lifeStateData.add(time);

            // Make sure every hero is present before starting
            for (int i = 0; i < numPlayers; i++) 
            {
                int herohandle = getEntityProperty(pr, "m_vecPlayerTeamData.%i.m_hSelectedHero", validIndices[i]);
                Entity heroEntity = ctx.getProcessor(Entities.class).getByHandle(herohandle);
                if (heroEntity == null)
                {
                    return;
                }
                else
                {
                    Integer life_state = getEntityProperty(heroEntity,"m_lifeState",null);
                    lifeStateData.add((float) life_state );
                }
            }
            if(save_full_res_lifestate == true)
            {
                write_life_state_to_cvs(lifeStateData); 
            }
            

            if ((tick % this.TICKS_PER_DATAPOINTS) != 0) // is it time for new data point
            {
                return;
            }

            // make sure all the towers are present
            if (latest_tower_entities.size() != 22)
            {
                return;
            }

            StringTable stEntityNames = ctx.getProcessor(StringTables.class).forName("EntityNames");
            List<Float> data_point = new ArrayList<Float>();
           

            addSingleFeature(data_point,time,"time");

            for (Integer i = 0; i < numPlayers; i++) 
            {
                String feature_name_prefix = "player_" + i.toString() + "_";

                // ###############################
                // PLAYER RESOURCES ATRIBUTES
                // ###############################

                for (String attribute : player_resources_attributes) 
                {
                    addEntityPropertyToFeatureVec(data_point,pr,attribute,validIndices[i],feature_name_prefix);
                }

                // #################
                // TEAM ATRIBUTES
                // #################

                int playerTeam = getEntityProperty(pr, "m_vecPlayerData.%i.m_iPlayerTeam", validIndices[i]);
                int teamSlot = getEntityProperty(pr, "m_vecPlayerTeamData.%i.m_iTeamSlot", validIndices[i]);
                Entity dataTeam = playerTeam == 2 ? rData : dData;
                if (teamSlot < 0) { System.out.println("ERROR negative team slot: " + teamSlot);}


                for (String attribute : team_attributes) 
                {
                    addEntityPropertyToFeatureVec(data_point,dataTeam,attribute,teamSlot,feature_name_prefix);
                }

                // #################
                // HERO ATRIBUTES
                // #################
                int herohandle = getEntityProperty(pr, "m_vecPlayerTeamData.%i.m_hSelectedHero", validIndices[i]);
                Entity heroEntity = ctx.getProcessor(Entities.class).getByHandle(herohandle);

                for (String attribute : hero_attributes) 
                {
                    addEntityPropertyToFeatureVec(data_point,heroEntity,attribute,null,feature_name_prefix);
                }

                // ##########################
                // HERO ABILITY ATTRIBUTES
                // ##########################
                for(Integer ability_index = 0; ability_index < this.NUM_ABILITIES; ability_index++ )
                {
                    String ability_feature_name_prefix = feature_name_prefix + "ability_" + ability_index.toString() + "_";

                    int abilityID = heroEntity.getProperty(String.format("m_hAbilities.%04d",ability_index));
                    Entity ability_entity = ctx.getProcessor(Entities.class).getByHandle(abilityID);

                    for (String attribute : hero_ability_attributes) 
                    {
                        addEntityPropertyToFeatureVec(data_point,ability_entity,attribute,null,ability_feature_name_prefix);
                    }
                    
                }

                // ########
                // ITEMS
                // ########
                List<Float> activatable_item_features = new ArrayList<Float>(Collections.nCopies(
                                                    op_activatable_items.length * 2 , 0.0f)); // is owned and cooldown
                List<Float> consumable_item_features = new ArrayList<Float>(Collections.nCopies(op_consumable_items.length, 0.0f));

                for (Integer item_i = 0; item_i < NUM_ITEMS; item_i++)
                {
                    int itemHandle = heroEntity.getProperty(String.format("m_hItems.%04d",item_i));
                    Entity item_entity = ctx.getProcessor(Entities.class).getByHandle(itemHandle);
                    if (item_entity != null)
                    {
                        Integer string_table_name_index = item_entity.getProperty("m_pEntity.m_nameStringableIndex");
                        String itemName = stEntityNames.getNameByIndex(string_table_name_index);

                        for(int item_list_i=0; item_list_i < op_activatable_items.length; item_list_i++)
                        {
                            String op_item_name = op_activatable_items[item_list_i];
                            if (itemName.contains(op_item_name))
                            {
                                activatable_item_features.set(item_list_i*2+0, 1.0f); // item is owned, set it to 1
                                Float item_cooldown = getEntityProperty(item_entity,"m_fCooldown",null);
                                activatable_item_features.set(item_list_i*2+1, item_cooldown);
                            }
                        }
                        for(int item_list_i=0; item_list_i < op_consumable_items.length; item_list_i++)
                        {
                            String op_item_name = op_consumable_items[item_list_i];
                            if (itemName.contains(op_item_name))
                            {
                                consumable_item_features.set(item_list_i, 1.0f);
                            }
                        }
                    }
                }
                for(int item_list_i=0; item_list_i < op_activatable_items.length; item_list_i++)
                {
                    addSingleFeature(data_point,activatable_item_features.get(item_list_i*2+0),feature_name_prefix+"item_owned_"+op_activatable_items[item_list_i]);
                    addSingleFeature(data_point,activatable_item_features.get(item_list_i*2+1),feature_name_prefix+"item_cooldown_"+op_activatable_items[item_list_i]);
                }
                for(int item_list_i=0; item_list_i < op_consumable_items.length; item_list_i++)
                {
                    addSingleFeature(data_point,consumable_item_features.get(item_list_i),feature_name_prefix+"consumable_"+op_consumable_items[item_list_i]);
                }

            } // end for each player


            // ##################
            // TOWER ATTRIBUTES
            // ##################
            for (Map.Entry<String,Entity> entry : latest_tower_entities.entrySet()) 
            {
                String tower_name = entry.getKey();
                Entity tower_entity = entry.getValue();

                for (String attribute : tower_attributes) 
                {
                    addEntityPropertyToFeatureVec(data_point,tower_entity,attribute,null,tower_name);
                }
            }

            // #############
            // SAVE DATA
            // #############
            if (this.feature_list_done == false)
            {
                System.out.println(feature_names.size());
                System.out.println(data_point.size());
                write_feature_names_to_cvs(feature_names);
            }
            this.feature_list_done = true;

            write_feature_vec_to_cvs(data_point);
        } 
    }



    void saveFeatureName(String feature_name,String label_prefix)
    {
        if(this.feature_list_done == false)
        {
            String full_feature_name = feature_name;
            if(label_prefix != null)
            {
                full_feature_name = label_prefix + feature_name;
            }
            //System.out.println(full_feature_name);
            this.feature_names.add(full_feature_name);
        }
    }

    public String ResolvePropertyName(String property, Integer idx)
    {
        if (idx != null) 
        {
            property = property.replace("%i", Util.arrayIdxToString(idx));
        }
        return property;
    }

    public void addIntEntityPropertyToFeatureVec(List<Float> feature_vec, Entity e, String property, Integer idx,String label_prefix)
    {
        try 
        {
            property = ResolvePropertyName(property,idx);
            saveFeatureName(property,label_prefix);
            FieldPath fp = e.getDtClass().getFieldPathForName(property);
            Integer val = e.getPropertyForFieldPath(fp);
            if (val == null)
            {
                feature_vec.add(0.0f);
            }
            else
            {
                feature_vec.add((float)val);
            }
        }
        catch (Exception ex) 
        {
            error_has_occured = true;
            error_set.add(property);
            //System.out.println("exception  " + property + "  " + ex.getMessage());
    	}
    }
    public void addBoolEntityPropertyToFeatureVec(List<Float> feature_vec, Entity e, String property, Integer idx,String label_prefix)
    {
        try 
        {
            property = ResolvePropertyName(property,idx);
            saveFeatureName(property,label_prefix);
            FieldPath fp = e.getDtClass().getFieldPathForName(property);
            Boolean val = e.getPropertyForFieldPath(fp);
            if (val == null)
            {
                feature_vec.add(0.0f);
            }
            else
            {
                feature_vec.add(val ? 1.0f : 0.0f);
            }
        }
        catch (Exception ex) 
        {
            //feature_vec.add(0.0f); // write 0 instead of missing data, or make them corrupt and ignore them?
            error_has_occured = true;
            error_set.add(property);
            //System.out.println("exception  " + property + "  " + ex.getMessage());
    	}
    }
    public void addFloatEntityPropertyToFeatureVec(List<Float> feature_vec, Entity e, String property, Integer idx,String label_prefix)
    {
        try 
        {
            property = ResolvePropertyName(property,idx);
            saveFeatureName(property,label_prefix);
            FieldPath fp = e.getDtClass().getFieldPathForName(property);
            Float val = e.getPropertyForFieldPath(fp);
            if (val == null)
            {
                feature_vec.add(0.0f);
            }
            else
            {
                feature_vec.add(val);
            }
        }
        catch (Exception ex) 
        {
            error_has_occured = true;
            error_set.add(property);
            //System.out.println("exception  " + property + "  " + ex.getMessage());
    	}
    }


    public void init_lifestate_cvs()
    {
        try
        {
            this.life_state_cvs_file.write("time,");
            for (Integer i = 0; i < 10; i++)
            {
                this.life_state_cvs_file.write("player_" + i.toString() + "_m_lifeState");
                if (i == 9)
                {
                    this.life_state_cvs_file.write("\n");
                }
                else
                {
                    this.life_state_cvs_file.write(",");
                }
            }
        }
        catch (Exception ex) 
        {
            error_has_occured = true;
            System.out.println("trouble writing to life_state cvs:  " + ex.getMessage());
        }
    }

    public void write_life_state_to_cvs(List<Float> life_state)
    {
        try
        {
            for (Integer i = 0; i < life_state.size(); i++)
            {
                this.life_state_cvs_file.write(life_state.get(i).toString());
                if (i != life_state.size()-1) // no comma after last element
                {
                    this.life_state_cvs_file.write(",");
                }
            }
            this.life_state_cvs_file.write("\n");
        }
        catch (Exception ex) 
        {
            error_has_occured = true;
            System.out.println("trouble writing to life_state cvs:  " + ex.getMessage());
        }
    }


    public void write_feature_names_to_cvs(List<String> feature_names)
    {
        try
        {
            for (Integer i = 0; i < feature_names.size(); i++)
            {
                this.cvs_file.write(feature_names.get(i));
                if (i != feature_names.size()-1) // no comma after last element
                {
                    this.cvs_file.write(",");
                }
            }
            this.cvs_file.write("\n");
        }
        catch (Exception ex) 
        {
            error_has_occured = true;
            System.out.println("trouble writing to cvs:  " + ex.getMessage());
        }
    }

    public void write_feature_vec_to_cvs(List<Float> feature_vec)
    {
        try
        {
            for (Integer i = 0; i < feature_vec.size(); i++)
            {
                this.cvs_file.write(feature_vec.get(i).toString());
                if (i != feature_vec.size()-1) // no comma after last element
                {
                    this.cvs_file.write(",");
                }
            }
            this.cvs_file.write("\n");
        }
        catch (Exception ex) 
        {
            error_has_occured = true;
            System.out.println("trouble writing to cvs:  " + ex.getMessage());
        }
    }

    public <T> T getEntityProperty(Entity e, String property, Integer idx) 
    {
        try 
        {
            if (e == null) 
            {
	            return null;
	        }
            if (idx != null) 
            {
	            property = property.replace("%i", Util.arrayIdxToString(idx));
	        }
	        FieldPath fp = e.getDtClass().getFieldPathForName(property);
	        return e.getPropertyForFieldPath(fp);
    	}
        catch (Exception ex) 
        {
            error_has_occured = true;
            error_set.add(property);
            //System.out.println("getEntityProperty exception  " + ex.getMessage());
    		return null;
    	}
    }

    public void run(String[] args) throws Exception 
    {
        // open cvs
        //Path p = Paths.get("/home/katona/workspace/esport/bad_dem.dem");
        //Path out_folder = Paths.get("/home/katona/workspace/esport/clarity-examples");

        Path p = Paths.get(args[0]);
        Path out_folder = Paths.get(args[1]);

        String match_name = p.getFileName().toString();
        match_name = match_name.replaceAll(".dem", "");

        this.cvs_file = new FileWriter( out_folder.toString() + "/" +  match_name + ".csv");
        this.life_state_cvs_file = new FileWriter(out_folder.toString() + "/" + match_name + "_life.csv");
        init_lifestate_cvs();

        long tStart = System.currentTimeMillis();
        new SimpleRunner(new MappedFileSource(args[0])).runWith(this);
        //new SimpleRunner(new MappedFileSource("/home/katona/workspace/esport/bad_dem.dem")).runWith(this);
        Long tMatch = System.currentTimeMillis() - tStart;
        System.out.println("parsing took: " + ((Double)(tMatch.doubleValue() / 1000.0)).toString());

        //System.out.println("Item set: " + Arrays.toString(item_names_set.toArray()));

        if (error_has_occured == true)
        {
            System.out.println("PARSING_ERROR: " + match_name);
            System.out.println("A PARSING_ERROR has occured, result is most likely corrupt: " + Arrays.toString(error_set.toArray()));
        }
        this.cvs_file.flush();
        this.cvs_file.close();
        this.life_state_cvs_file.flush();
        this.life_state_cvs_file.close();
    }


    public static void main(String[] args) throws Exception {
        try {
            //BitStreamImplementations.implementation = 1;
            //System.out.println("press key to start"); System.in.read();
            new Main().run(args);
        } catch (Exception e) {
            Thread.sleep(200);
            throw e;
        }
    }
}
