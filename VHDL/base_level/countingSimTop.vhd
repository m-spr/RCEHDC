-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;

    USE std.textio.ALL;

ENTITY countingSimTop IS
    GENERIC (n           : INTEGER := 10; --; 	-- bit-widths of memory pointer, counter and etc,,, 
             d           : INTEGER := 10; -- number of confComp module
             z           : INTEGER := 0;  -- zeropadding to 2** for RSA 
             classNumber : INTEGER := 10; ---- class number --- for memory image
             logInNum    : INTEGER := 3;
             dimensionSize: Integer := 1000); -- MuxCell, ceilingLOG2(#popCounters)
    PORT (
        clk, rst, run : IN  STD_LOGIC;
        hv            : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
        done          : OUT STD_LOGIC;
        pointer       : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
        dout          : OUT STD_LOGIC_VECTOR(classNumber * (n + logInNum) - 1 DOWNTO 0);

        update_valid : IN STD_LOGIC;
        update_done  : OUT STD_LOGIC;
        updated_truth           : IN std_logic_vector (999 downto 0);
        updated_prediction     : IN std_logic_vector (999 downto 0);
        ground_truth            : IN integer;
        predicted_label         : IN integer

    );
END ENTITY countingSimTop;

ARCHITECTURE behavioral OF countingSimTop IS

    COMPONENT countingSim IS
        GENERIC (n           : INTEGER := 10; --; 	-- bit-widths of memory pointer, counter and etc,,, 
                 d           : INTEGER := 10; -- number of confComp module
                 z           : INTEGER := 0;  -- zeropadding to 2** for RSA 
                 classNumber : INTEGER := 10; ---- class number --- for memory image
                 logInNum    : INTEGER := 3); -- MuxCell, ceilingLOG2(#popCounters OR d)
        PORT (
            clk, rst, run, done                      : IN  STD_LOGIC; ---- run shuld be always '1' during calculation --- ctrl ---- 
            reg1Update, reg1rst, reg2Update, reg2rst : IN  STD_LOGIC; ---- run shuld be always '1' during calculation --- ctrl ---- 
            muxSel                                   : IN  STD_LOGIC_VECTOR(logInNum DOWNTO 0);
            hv                                       : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            CHV                                      : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
            pointer                                  : IN  STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
            dout                                     : OUT STD_LOGIC_VECTOR(n + logInNum - 1 DOWNTO 0)
        );
    END COMPONENT countingSim;

    COMPONENT SeqAdderCtrl IS
        GENERIC (ceilingLogPop : INTEGER := 3; -- ceilingLOG2(#popCounters)
                 nPop          : INTEGER := 8); -- #popCounters
        PORT (
            clk, rst            : IN  STD_LOGIC;
            run                 : IN  STD_LOGIC;
            reg1Update, reg1rst : OUT STD_LOGIC;
            reg2Update, reg2rst : OUT STD_LOGIC;
            muxSel              : OUT STD_LOGIC_VECTOR(ceilingLogPop DOWNTO 0)
        );
    END COMPONENT SeqAdderCtrl;

    COMPONENT countingSimCtrl IS
        GENERIC (n : INTEGER := 10); --- bit pointer to memory
        PORT (
            clk, rst     : IN  STD_LOGIC;
            run          : IN  STD_LOGIC;
            runOut, done : OUT STD_LOGIC;
            pointer      : OUT STD_LOGIC_VECTOR(n - 1 DOWNTO 0) --- As of now only support up to 16 classes so 4'bits 
        );
    END COMPONENT countingSimCtrl;

    --new memory signals
    TYPE CHV_memory IS ARRAY (classNumber - 1 DOWNTO 0) OF std_logic_vector((2 ** (n)) * d - 1 DOWNTO 0);
    TYPE CHV_memory_tosim IS ARRAY (classNumber - 1 DOWNTO 0) OF std_logic_vector(d - 1 DOWNTO 0);
    SIGNAL CHV_TO_OUT : CHV_memory_tosim;

    SIGNAL CHV : CHV_memory;
    FILE CHV_file : text OPEN read_mode IS "CHV_img.mif"; -- Specify your file name

    SIGNAL dones, runOut                            : STD_LOGIC; ---- run shuld be always '1' during calculation --- ctrl ---- 
    SIGNAL reg1Update, reg1rst, reg2Update, reg2rst : STD_LOGIC; ---- run shuld be always '1' during calculation --- ctrl ---- 
    SIGNAL muxSel                                   : STD_LOGIC_VECTOR(logInNum DOWNTO 0);
    SIGNAL point                                    : STD_LOGIC_VECTOR(n - 1 DOWNTO 0);
    SIGNAL allZeros                                 : std_logic_vector((2 ** (n)) * d - 1 DOWNTO 0) := (OTHERS => '0');
    SIGNAL padded_updated_truth                        : std_logic_vector ((2 ** (n) * d) - 1 DOWNTO 0);
    SIGNAL padded_updated_prediction                   : std_logic_vector ((2 ** (n) * d) - 1 DOWNTO 0);

    ATTRIBUTE MARK_DEBUG               : string;
    ATTRIBUTE MARK_DEBUG OF CHV        : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF CHV_TO_OUT : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF update_valid : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF update_done : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF padded_updated_truth : SIGNAL IS "TRUE";
    ATTRIBUTE MARK_DEBUG OF padded_updated_prediction : SIGNAL IS "TRUE";

BEGIN
    padded_updated_truth <= allZeros((2 ** (n)) * d - 1 - dimensionSize DOWNTO 0) & updated_truth;
    padded_updated_prediction <= allZeros((2 ** (n)) * d - 1 - dimensionSize DOWNTO 0) & updated_prediction;

    PROCESS (clk)
        VARIABLE mif_line    : line;
        VARIABLE temp_bv     : bit_vector((2 ** (n)) * d - 1 DOWNTO 0);
        VARIABLE initialized : boolean := false;
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                update_done <= '0';
                -- Initialize from file
                IF NOT initialized THEN
                    FOR i IN 0 TO classNumber - 1 LOOP
                        IF NOT endfile(CHV_file) THEN
                            readline(CHV_file, mif_line);
                            read(mif_line, temp_bv);
                            CHV(i) <= to_stdlogicvector(temp_bv);
                        ELSE
                            CHV(i) <= (OTHERS => '0');
                        END IF;
                    END LOOP;
                    initialized := true;
                END IF;
            ELSIF update_valid = '1' THEN
                -- Update CHV vectors with learning results
                CHV(ground_truth) <= padded_updated_truth;
                CHV(predicted_label) <= padded_updated_prediction;
                update_done <= '1';
            ELSE
                update_done <= '0';
            END IF;
        END IF;
    END PROCESS;

    concatECC: FOR I IN classNumber - 1 DOWNTO 0 GENERATE
        classesECC: FOR k IN d - 1 DOWNTO 0 GENERATE
            CHV_TO_OUT(I)(k) <= CHV(I)(to_integer(unsigned(point)) + (2 ** (n)) * k);
        END GENERATE classesECC;
    END GENERATE concatECC;

    AdderCtrl: SeqAdderCtrl
        GENERIC MAP (logInNum,
                     d)
        PORT MAP (
            clk, rst,
            dones,
            reg1Update, reg1rst,
            reg2Update, reg2rst,
            muxSel
        );

    countSimArr: FOR I IN classNumber - 1 DOWNTO 0 GENERATE
        comp: countingSim
            GENERIC MAP (n, d, z, I, logInNum)
            PORT MAP (
                clk, rst, runOut, dones,
                reg1Update, reg1rst, reg2Update, reg2rst,
                muxSel,
                hv, CHV_TO_OUT(I),
                point,
                dout(((I + 1) * (n + logInNum)) - 1 DOWNTO ((I) * (n + logInNum)))
            );
    END GENERATE countSimArr;

    CompCtrl: countingSimCtrl
        GENERIC MAP (n)
        PORT MAP (
            clk, rst,
            run,
            runOut, dones,
            point
        );

    pointer <= point;
    done    <= reg2Update;

END ARCHITECTURE behavioral;